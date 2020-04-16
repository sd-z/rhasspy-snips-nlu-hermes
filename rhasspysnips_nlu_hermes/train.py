"""Training methods for Snips NLU."""
import io

import networkx as nx
import rhasspynlu


def write_dataset(dataset_file, sentences_dict, slots_dict):
    """Write intents/entities YAML dataset."""

    # Parse sentences and convert to graph
    with io.StringIO() as ini_file:
        # Join as single ini file
        for lines in sentences_dict.values():
            print(lines, file=ini_file)
            print("", file=ini_file)

        # Parse JSGF sentences
        intents = rhasspynlu.parse_ini(ini_file.getvalue())

    # Split into sentences and rule/slot replacements
    sentences, replacements = rhasspynlu.ini_jsgf.split_rules(intents)

    for intent_sentences in sentences.values():
        for sentence in intent_sentences:
            rhasspynlu.jsgf.walk_expression(
                sentence, rhasspynlu.number_range_transform, replacements
            )

    # Convert to directed graph *without* expanding slots
    # (e.g., $rhasspy/number)
    intent_graph = rhasspynlu.sentences_to_graph(
        sentences, replacements=replacements, expand_slots=False
    )

    # Get start/end nodes for graph
    start_node, end_node = rhasspynlu.jsgf_graph.get_start_end_nodes(intent_graph)
    assert (start_node is not None) and (
        end_node is not None
    ), "Missing start/end node(s)"

    # Walk first layer of edges with intents
    for _, intent_node, edge_data in intent_graph.edges(start_node, data=True):
        intent_name: str = edge_data["olabel"][9:]

        # New intent
        print("---", file=dataset_file)
        print("type: intent", file=dataset_file)
        print("name:", quote(intent_name), file=dataset_file)
        print("utterances:", file=dataset_file)

        # Get all paths through the graph (utterances)
        paths = nx.all_simple_paths(intent_graph, intent_node, end_node)
        for path in paths:
            utterance = []
            entity_name = None
            slot_name = None
            slot_value = None

            # Walk utterance edges
            for from_node, to_node in rhasspynlu.utils.pairwise(path):
                edge_data = intent_graph.edges[(from_node, to_node)]
                ilabel = edge_data.get("ilabel")
                olabel = edge_data.get("olabel")
                if olabel:
                    if olabel.startswith("__begin__"):
                        slot_name = olabel[9:]
                        entity_name = None
                        slot_value = ""
                    elif olabel.startswith("__end__"):
                        if entity_name == "rhasspy/number":
                            # Transform to Snips number
                            entity_name = "snips/number"
                        elif not entity_name:
                            # Collect actual value
                            entity_name = slot_name
                            if slot_name not in slots_dict:
                                slots_dict[slot_name] = set()

                            slots_dict[slot_name].add(slot_value.strip())

                        # Reference slot/entity (values will be added later)
                        utterance.append(f"[{slot_name}:{entity_name}]")

                        # Reset current slot/entity
                        entity_name = None
                        slot_name = None
                        slot_value = None
                    elif olabel.startswith("__source__"):
                        # Use Rhasspy slot name as entity
                        entity_name = olabel[10:]

                if ilabel:
                    if slot_name and not entity_name:
                        # Add to current slot/entity value
                        slot_value += ilabel + " "
                    else:
                        # Add directly to utterance
                        utterance.append(ilabel)

            if utterance:
                # Write utterance
                print("  -", quote(" ".join(utterance)), file=dataset_file)

        print("", file=dataset_file)

    # Write entities
    for slot_name, values in slots_dict.items():
        if slot_name.startswith("$"):
            # Remove arguments and $
            slot_name = slot_name.split(",")[0][1:]

        # Skip numbers
        if slot_name in {"rhasspy/number"}:
            # Should have been converted already to snips/number
            continue

        print("---", file=dataset_file)
        print("type: entity", file=dataset_file)
        print("name:", quote(slot_name), file=dataset_file)
        print("values:", file=dataset_file)
        for value in values:
            print("  -", quote(value), file=dataset_file)

        print("", file=dataset_file)


def quote(s):
    """Surround with quotes for YAML."""
    return f'"{s}"'

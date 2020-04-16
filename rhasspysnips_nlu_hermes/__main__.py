"""Hermes MQTT service for Snips NLU"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt
import rhasspyhermes.cli as hermes_cli

from . import NluHermesMqtt

_LOGGER = logging.getLogger("rhasspysnips_nlu_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    parser = argparse.ArgumentParser(prog="rhasspy-snips-nlu-hermes")
    parser.add_argument(
        "--language",
        required=True,
        help="Snips language (de, en, es, fr, it, ja, ko, pt_br, pt_pt, zh)",
    )
    parser.add_argument("--engine-path", help="Path to read/write trained engine")
    parser.add_argument(
        "--dataset-path", help="Path to write YAML dataset during training"
    )
    parser.add_argument(
        "--casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for input text (default: ignore)",
    )
    parser.add_argument(
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite Snips engine configuration during training",
    )

    hermes_cli.add_hermes_args(parser)

    args = parser.parse_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    # Convert to Paths
    if args.engine_path:
        args.engine_path = Path(args.engine_path)

    if args.dataset_path:
        args.dataset_path = Path(args.dataset_path)

    # Listen for messages
    client = mqtt.Client()
    hermes = NluHermesMqtt(
        client,
        snips_language=args.language,
        engine_path=args.engine_path,
        dataset_path=args.dataset_path,
        word_transform=get_word_transform(args.casing),
        no_overwrite_train=args.no_overwrite_train,
        site_ids=args.site_id,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

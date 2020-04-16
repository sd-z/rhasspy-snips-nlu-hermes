"""Hermes MQTT server for Snips NLU"""
import io
import logging
import time
import typing
from pathlib import Path

from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluIntentParsed,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from snips_nlu import SnipsNLUEngine
from snips_nlu.dataset import Dataset
from snips_nlu.default_configs import DEFAULT_CONFIGS

from .train import write_dataset

_LOGGER = logging.getLogger("rhasspysnips_nlu_hermes")

# -----------------------------------------------------------------------------


class NluHermesMqtt(HermesClient):
    """Hermes MQTT server for Snips NLU."""

    def __init__(
        self,
        client,
        snips_language: str,
        engine_path: typing.Optional[Path] = None,
        dataset_path: typing.Optional[Path] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        no_overwrite_train: bool = False,
        site_ids: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__("rhasspysnips_nlu_hermes", client, site_ids=site_ids)

        self.subscribe(NluQuery, NluTrain)

        self.snips_language = snips_language
        self.engine_path = engine_path
        self.dataset_path = dataset_path

        self.word_transform = word_transform

        self.no_overwrite_train = no_overwrite_train

        self.engine: typing.Optional[SnipsNLUEngine] = None

    # -------------------------------------------------------------------------

    async def handle_query(
        self, query: NluQuery
    ) -> typing.AsyncIterable[
        typing.Union[
            NluIntentParsed,
            typing.Tuple[NluIntent, TopicArgs],
            NluIntentNotRecognized,
            NluError,
        ]
    ]:
        """Do intent recognition."""
        original_input = query.input

        try:
            self.maybe_load_engine()
            assert self.engine, "Snips engine not loaded. You may need to train."

            input_text = query.input

            # Fix casing for output event
            if self.word_transform:
                input_text = self.word_transform(input_text)

            # Do parsing
            result = self.engine.parse(input_text, query.intent_filter)
            intent_name = result.get("intent", {}).get("intentName")

            if result and intent_name:
                slots = [
                    Slot(
                        slot_name=s["slotName"],
                        entity=s["entity"],
                        value=s["value"],
                        raw_value=s["rawValue"],
                        range=SlotRange(
                            start=s["range"]["start"], end=s["range"]["end"]
                        ),
                    )
                    for s in result.get("slots", [])
                ]

                # intentParsed
                yield NluIntentParsed(
                    input=query.input,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                    intent=Intent(intent_name=intent_name, confidence_score=1.0),
                    slots=slots,
                )

                # intent
                yield (
                    NluIntent(
                        input=query.input,
                        id=query.id,
                        site_id=query.site_id,
                        session_id=query.session_id,
                        intent=Intent(intent_name=intent_name, confidence_score=1.0),
                        slots=slots,
                        asr_tokens=[NluIntent.make_asr_tokens(query.input.split())],
                        raw_input=original_input,
                        wakeword_id=query.wakeword_id,
                    ),
                    {"intent_name": intent_name},
                )
            else:
                # Not recognized
                yield NluIntentNotRecognized(
                    input=query.input,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                )
        except Exception as e:
            _LOGGER.exception("handle_query")
            yield NluError(
                site_id=query.site_id,
                session_id=query.session_id,
                error=str(e),
                context=original_input,
            )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: NluTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Transform sentences/slots into Snips NLU training dataset."""
        try:
            assert train.sentences, "No training sentences"

            start_time = time.perf_counter()

            if self.dataset_path:
                # Write to user path
                self.dataset_path.parent.mkdir(exist_ok=True)
                dataset_file = open(self.dataset_path, "w+")
            else:
                # Write to buffer
                dataset_file = io.StringIO()

            with dataset_file:
                # Generate dataset
                write_dataset(dataset_file, train.sentences or {}, train.slots or {})

                # Train engine
                dataset_file.seek(0)
                dataset = Dataset.from_yaml_files(self.snips_language, [dataset_file])

                new_engine = self.get_empty_engine()
                new_engine = new_engine.fit(dataset)

            end_time = time.perf_counter()

            _LOGGER.debug("Trained Snips engine in %s second(s)", end_time - start_time)
            self.engine = new_engine

            if not self.no_overwrite_train and self.engine_path:
                # Save engine
                self.engine_path.parent.mkdir(exist_ok=True)
                self.engine.persist(self.engine_path)
                _LOGGER.debug("Saved Snips engine to %s", self.engine_path)

            yield (NluTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield NluError(
                site_id=site_id, session_id=train.id, error=str(e), context=train.id
            )

    def get_empty_engine(self):
        """Load Snips engine configured for specific language."""
        assert (
            self.snips_language in DEFAULT_CONFIGS
        ), f"Snips language not supported: {self.snips_language}"

        _LOGGER.debug("Creating empty Snips engine (language=%s)", self.snips_language)
        return SnipsNLUEngine(config=DEFAULT_CONFIGS[self.snips_language])

    def maybe_load_engine(self):
        """Load Snips engine if not already loaded."""
        if self.engine:
            # Already loaded
            return

        if self.engine_path and self.engine_path.exists():
            _LOGGER.debug("Loading Snips engine from %s", self.engine_path)
            self.engine = SnipsNLUEngine.from_path(self.engine_path)

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        if isinstance(message, NluQuery):
            async for query_result in self.handle_query(message):
                yield query_result
        elif isinstance(message, NluTrain):
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)

"""Generative models from an internally hosted vLLM OpenAI server."""

import asyncio
import collections.abc as c
import logging
import typing as t
from functools import cached_property, partial
from time import sleep

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.types.utils import ChoiceLogprobs
from tqdm.asyncio import tqdm as tqdm_async

from ..constants import MAX_LOGPROBS, REASONING_MAX_TOKENS, TASKS_USING_JSON
from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import (
    InvalidBenchmark,
    InvalidModel,
    NeedsAdditionalArgument,
    NeedsExtraInstalled,
    NeedsEnvironmentVariable,
)
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
)
from ..tokenization_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from ..utils import (
    add_semaphore_and_catch_exception,
    log_once,
    safe_run,
)
from .base import BenchmarkModule

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from litellm.types.utils import ModelResponse
    from transformers.trainer import Trainer

logger = logging.getLogger("euroeval")


class LiteLLMInternalModel(BenchmarkModule):
    """A generative model from an internally hosted vLLM OpenAI server."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = True  # High priority to be selected over HF module

    _handleable_exceptions = (
        BadRequestError,
        RateLimitError,
        APIError,
        APIConnectionError,
        Timeout,
        ServiceUnavailableError,
        InternalServerError,
        SystemError,
        AuthenticationError,
    )

    # Supported task groups for this module (keeping it simple initially)
    SUPPORTED_TASK_GROUPS = {
        TaskGroup.SEQUENCE_CLASSIFICATION,
        TaskGroup.TEXT_TO_TEXT,
        TaskGroup.QUESTION_ANSWERING,
    }

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        # Validate that the task is supported
        if dataset_config.task.task_group not in self.SUPPORTED_TASK_GROUPS:
            raise InvalidModel(
                f"Task group {dataset_config.task.task_group.value} is not supported "
                f"by the LiteLLM Internal module. Supported task groups: "
                f"{[tg.value for tg in self.SUPPORTED_TASK_GROUPS]}"
            )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokenizer=None,
            generative_type=self.generative_type,
        )

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        # For internal vLLM server, assume instruction-tuned models
        type_ = GenerativeType.INSTRUCTION_TUNED

        log_once(
            f"Detected generative type {type_.name!r} for model "
            f"{self.model_config.model_id!r}",
            level=logging.DEBUG,
        )
        return type_

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        assert "messages" in inputs, "The input must contain a 'messages' key."
        conversations: list[list[litellm.AllMessageValues]] = inputs["messages"]

        # Get the mapping from labels to the first token in the label
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokenizer=None,
            generative_type=self.generative_type,
        )

        # Set the core generation arguments
        generation_kwargs: dict[str, t.Any] = dict(
            model=self.model_config.model_id,
            max_completion_tokens=self.dataset_config.max_generated_tokens,
            stop=[],
            temperature=0.0,
            seed=4242,
            api_key=self.benchmark_config.api_key,
            api_base=self.benchmark_config.api_base,
            api_version=self.benchmark_config.api_version,
            max_retries=3,
        )

        # Set up the `response_format` generation argument if we are dealing with a task
        # using structured generation
        if self.dataset_config.task in TASKS_USING_JSON:
            # Sanity check that "JSON" is included in the prompt
            for conversation in conversations:
                if not conversation:
                    raise InvalidBenchmark(
                        "Encountered an empty conversation in 'messages'."
                    )
                last_message = conversation[-1]
                assert isinstance(last_message, dict), (
                    f"Expected dict message, got {type(last_message)}"
                )
                assert "content" in last_message, (
                    "Expected 'content' key in the last message of the conversation."
                )
                assert isinstance(last_message["content"], str), (
                    "Expected 'content' to be a string."
                )
                assert "json" in last_message["content"].lower(), (
                    "Prompt must contain 'json' for JSON tasks."
                )

            # Use vanilla JSON format for internal server
            generation_kwargs["response_format"] = dict(type="json_object")
            log_once(
                "Enabling structured JSON generation for internal vLLM server "
                f"{self.model_config.model_id!r}",
                level=logging.DEBUG,
            )

        # Handle logprobs for classification tasks
        if self.buffer["first_label_token_mapping"]:
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = MAX_LOGPROBS

        # Drop generation kwargs that are not supported by the model
        litellm.drop_params = True

        all_responses: dict[int, "ModelResponse"] = {}
        conversations_to_run: list[tuple[int, list[litellm.AllMessageValues]]] = list(
            enumerate(conversations)
        )
        for attempt in range(num_attempts := 10):
            if not conversations_to_run:
                break

            batch_indices, batch_conversations = zip(*conversations_to_run)
            successes, failures = safe_run(
                self._generate_async(
                    model_id=self.model_config.model_id,
                    conversations=list(batch_conversations),
                    **generation_kwargs,
                )
            )

            # Store the successful model outputs
            for idx, response in successes:
                orig_idx = batch_indices[idx]
                all_responses[orig_idx] = response

            # If all requests were successful, break
            if not failures:
                conversations_to_run = []
                break

            # Put the failed requests back in the queue to try again
            conversations_to_run = [
                (batch_indices[idx], conversations[batch_indices[idx]])
                for idx, _ in failures
            ]
            logger.debug(
                f"Attempt {attempt + 1:,}/{num_attempts:,}: retrying "
                f"{len(conversations_to_run):,} failed message(s)"
            )

            # Attempt to handle the exceptions
            for _, error in failures:
                self._handle_exception(error=error, generation_kwargs=generation_kwargs)

            # Sleep for a second to avoid pinging the API server too quickly
            sleep(1)
        else:
            raise InvalidBenchmark(
                message=f"Failed to generate text, after {num_attempts:,} attempts."
            )

        # Extract the generations from the model output
        ordered_responses = [all_responses[i] for i in range(len(conversations))]
        model_output = self._create_model_output(
            model_responses=ordered_responses, model_id=self.model_config.model_id
        )

        if len(conversations) != len(model_output.sequences):
            raise InvalidBenchmark(
                f"Number of model inputs ({len(conversations):,}) does not match the "
                f"number of model outputs ({len(model_output.sequences):,})."
            )

        return model_output

    def _handle_exception(
        self, error: Exception, generation_kwargs: dict[str, t.Any]
    ) -> None:
        """Handle an exception from the model.

        Args:
            error:
                The exception to handle.
            generation_kwargs:
                The generation kwargs to pass to the model.
        """
        error_msg = str(error).lower()
        model_id = self.model_config.model_id

        # Error messages that we want to catch and handle
        stop_messages = ["stop_sequences", "'stop' is not supported with this model"]
        logprobs_messages = [
            "you are not allowed to request logprobs",
            "you've reached the maximum number of requests with logprobs",
            "logprobs is not supported",
            "logprobs is not enabled",
        ]
        temperature_messages = [
            "'temperature' is not supported with this model.",
            "temperature is not supported with this model",
        ]
        temperature_must_be_one_messages = [
            "`temperature` may only be set to 1",
            "'temperature' does not support 0.0 with this model. Only the default "
            "(1) value is supported",
        ]

        if any(msg.lower() in error_msg for msg in stop_messages):
            log_once(
                f"The model {model_id!r} does not support "
                "stop sequences, so disabling them.",
                level=logging.DEBUG,
            )
            generation_kwargs["stop"] = None
            return
        elif any(msg.lower() in error_msg for msg in logprobs_messages):
            log_once(
                f"The model {model_id!r} does not support logprobs, so disabling it.",
                level=logging.DEBUG,
            )
            generation_kwargs.pop("logprobs", None)
            generation_kwargs.pop("top_logprobs", None)
            return
        elif any(msg.lower() in error_msg for msg in temperature_messages):
            log_once(
                f"The model {model_id!r} does not support "
                "temperature, so disabling it.",
                level=logging.DEBUG,
            )
            generation_kwargs.pop("temperature", None)
            return
        elif any(msg.lower() in error_msg for msg in temperature_must_be_one_messages):
            log_once(
                f"The model {model_id!r} requires "
                "temperature to be set to 1, so setting it.",
                level=logging.DEBUG,
            )
            generation_kwargs["temperature"] = 1.0
            return
        elif isinstance(
            error, (Timeout, ServiceUnavailableError, InternalServerError, SystemError)
        ):
            logger.debug(
                f"Service temporarily unavailable. The error message was: {error}. "
                f"Retrying in 5 seconds..."
            )
            sleep(5)
            return
        elif isinstance(error, (APIConnectionError, OSError)):
            raise InvalidBenchmark(
                f"Encountered {type(error)} during generation: {error}."
            )

        if isinstance(error, RateLimitError):
            raise InvalidModel(
                f"You have encountered your rate limit for model {model_id!r}. "
                "Skipping."
            )

        if isinstance(error, AuthenticationError):
            raise NeedsAdditionalArgument(
                cli_argument="--api-key",
                script_argument="api_key=<your-api-key>",
                run_with_cli=self.benchmark_config.run_with_cli,
            )

        raise InvalidBenchmark(
            f"Failed to generate text. The error message was: {error}"
        )

    async def _generate_async(
        self,
        model_id: str,
        conversations: list[list[litellm.AllMessageValues]],
        **generation_kwargs,
    ) -> tuple[list[tuple[int, "ModelResponse"]], list[tuple[int, Exception]]]:
        """Generate outputs from the model asynchronously.

        Args:
            model_id:
                The ID of the model to use for generation.
            conversations:
                The conversations to pass to the model.
            **generation_kwargs:
                Additional generation arguments to pass to the model.

        Returns:
            A tuple (successes, failures), each being a list of tuples (idx, content),
            where the `idx` corresponds to the index of `conversations`, and `content`
            is either the model response or an Exception.
        """
        # Create a LiteLLM router for the internal server
        print(generation_kwargs)
        router = litellm.Router(
            model_list=[
                dict(
                    model_name=self.model_config.model_id,
                    litellm_params=generation_kwargs,
                )
            ]
        )

        # Get the LLM generations asynchronously
        max_concurrent_calls = 20
        semaphore = asyncio.Semaphore(max_concurrent_calls)
        requests = [
            add_semaphore_and_catch_exception(
                router.acompletion(model=model_id, messages=conversation),
                semaphore=semaphore,
            )
            for conversation in conversations
        ]
        responses = await tqdm_async.gather(*requests, leave=False)

        # Separate the successful responses from the failed ones
        successes = [
            (idx, response)
            for idx, response in enumerate(responses)
            if not isinstance(response, Exception)
        ]
        failures = [
            (idx, response)
            for idx, response in enumerate(responses)
            if isinstance(response, Exception)
        ]

        # Close connections
        for request in requests:
            if hasattr(request, "close"):
                request.close()

        return successes, failures

    @staticmethod
    def _create_model_output(
        model_responses: list["ModelResponse"], model_id: str
    ) -> GenerativeModelOutput:
        """Create a GenerativeModelOutput object from a list of ModelResponse objects.

        Args:
            model_responses:
                The list of ModelResponse objects to create the GenerativeModelOutput
                object from.
            model_id:
                The ID of the model.

        Returns:
            A GenerativeModelOutput object.
        """
        sequences = []
        scores = []
        for model_response in model_responses:
            if not model_response.choices:
                sequences.append("")
                logger.warning(
                    f"The model {model_id!r} did not end up "
                    "generating any text. This is likely because the model ran "
                    "out of tokens while reasoning. Returning an empty string."
                )
                continue

            model_response_choices = model_response.choices[0]
            assert isinstance(model_response_choices, litellm.Choices)
            generated_message: litellm.Message = model_response_choices.message
            generation_output = generated_message.content or ""
            generation_output = generation_output.strip()

            # Structure the model output as a GenerativeModelOutput object
            sequences.append(generation_output)
            if hasattr(model_response_choices, "logprobs"):
                logprobs_obj = model_response_choices.logprobs
                if isinstance(logprobs_obj, ChoiceLogprobs):
                    logprobs_list: list[list[tuple[str, float]]] = [
                        [
                            (top_logprob.token, top_logprob.logprob)
                            for top_logprob in content.top_logprobs
                        ]
                        for content in model_response_choices.logprobs.content or list()
                    ]
                    scores.append(logprobs_list)
                else:
                    log_once(
                        "The logprobs object is malformed, so we won't use logprobs to "
                        "determine the labels.",
                        level=logging.WARNING,
                    )

        if not sequences:
            logger.warning(
                "No sequences were generated by the model "
                f"{model_id!r}. This may be due to the "
                "model running out of tokens or an issue with the input data. "
                "Returning an empty GenerativeModelOutput."
            )
            return GenerativeModelOutput(sequences=[], scores=None)

        if scores and len(sequences) != len(scores):
            raise InvalidBenchmark(
                "Sequences and scores must have the same length. "
                f"Got {len(sequences)} sequences and {len(scores)} scores."
            )

        return GenerativeModelOutput(
            sequences=sequences, scores=scores if scores else None
        )

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        # For internal vLLM server, we don't have access to model metadata
        # Return -1 to indicate unknown
        return -1

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        # For internal vLLM server, we don't have access to model metadata
        # Return -1 to indicate unknown
        return -1

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        # For internal vLLM server, we don't have access to model metadata
        # Return -1 to indicate unknown
        return -1

    @property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for LiteLLM Internal models."
        )

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        match self.dataset_config.task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                return partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                    first_label_token_mapping=self.buffer["first_label_token_mapping"],
                )
            case TaskGroup.TEXT_TO_TEXT:
                return text_to_text.extract_labels_from_generation
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.extract_labels_from_generation
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for LiteLLM Internal models."
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        # This method is called during model discovery, but we don't have access to
        # the ModelConfig yet. We'll rely on the model_loading.py logic to select
        # this module based on the internal_server flag.
        # For now, return False to let other modules handle the model discovery.
        return False

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "")
        return ModelConfig(
            model_id=model_id,
            revision=revision,
            task="text-generation",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.LITELLM,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=None,  # No local caching for internal server
            adapter_base_model_id=None,
            internal_server=True,  # Mark this as an internal server model
        )

    def prepare_dataset(
        self, dataset: "DatasetDict", task: Task, itr_idx: int
    ) -> "DatasetDict":
        """Prepare the dataset for the model.

        This includes things like tokenisation.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The prepared dataset.
        """
        if task.task_group == TaskGroup.QUESTION_ANSWERING:
            dataset = dataset.map(
                lambda examples: dict(
                    label=[
                        dict(
                            id=id,
                            answers=dict(
                                answer_start=answer_dct["answer_start"],
                                text=[
                                    answer_text.lower()
                                    for answer_text in answer_dct["text"]
                                ],
                            ),
                        )
                        for id, answer_dct in zip(examples["id"], examples["answers"])
                    ]
                ),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        if self.benchmark_config.few_shot:
            few_shot_examples = extract_few_shot_examples(
                dataset=dataset, dataset_config=self.dataset_config, itr_idx=itr_idx
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(
                apply_prompt,
                few_shot_examples=few_shot_examples,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                instruction_model=True,
                always_populate_text_field=False,
                tokenizer=None,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

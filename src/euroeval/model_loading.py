"""Functions related to the loading of models."""

import typing as t

from .benchmark_modules import (
    FreshEncoderModel,
    HuggingFaceEncoderModel,
    LiteLLMModel,
    LiteLLMInternalModel,
    VLLMModel,
)
from .enums import InferenceBackend, ModelType
from .exceptions import InvalidModel

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def load_model(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> "BenchmarkModule":
    """Load a model.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The model.
    """
    # If use_internal_server is set, always use LiteLLMInternalModel for generative models
    if getattr(benchmark_config, "use_internal_server", False) and model_config.model_type == ModelType.GENERATIVE:
        from .benchmark_modules import LiteLLMInternalModel
        model_class = LiteLLMInternalModel
    else:
        model_class: t.Type[BenchmarkModule]
        match (model_config.model_type, model_config.inference_backend, model_config.fresh):
            case (ModelType.GENERATIVE, InferenceBackend.VLLM, False):
                model_class = VLLMModel
            case (ModelType.ENCODER, InferenceBackend.TRANSFORMERS, False):
                model_class = HuggingFaceEncoderModel
            case (ModelType.GENERATIVE, InferenceBackend.LITELLM, False):
                if model_config.internal_server:
                    model_class = LiteLLMInternalModel
                else:
                    model_class = LiteLLMModel
            case (ModelType.ENCODER, InferenceBackend.TRANSFORMERS, True):
                model_class = FreshEncoderModel
            case (_, _, True):
                raise InvalidModel(
                    "Cannot load a freshly initialised model with the model type "
                    f"{model_config.model_type!r} and inference backend "
                    f"{model_config.inference_backend!r}."
                )
            case _:
                raise InvalidModel(
                    f"Cannot load model with model type {model_config.model_type!r} and "
                    f"inference backend {model_config.inference_backend!r}."
                )

    model = model_class(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    return model

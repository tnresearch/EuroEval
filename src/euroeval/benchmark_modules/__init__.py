"""The different types of modules that can be benchmarked."""

from .base import BenchmarkModule
from .fresh import FreshEncoderModel
from .hf import HuggingFaceEncoderModel
from .litellm import LiteLLMModel
from .litellm_internal import LiteLLMInternalModel
from .vllm import VLLMModel

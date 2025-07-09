#!/usr/bin/env python3
"""Simple test to verify internal vLLM module can call inference server."""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from euroeval.benchmark_modules.litellm_internal import LiteLLMInternalModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import ModelType, InferenceBackend, TaskGroup
from euroeval.data_models import Task
from euroeval.tasks import SUMM
from euroeval.languages import EN


def test_internal_vllm_server():
    """Test that the internal vLLM module can call the inference server."""
    
    # Configuration for your internal vLLM server
    benchmark_config = BenchmarkConfig(
        model_languages=[EN],
        dataset_languages=[EN],
        tasks=[SUMM],
        datasets=["test_dataset"],
        batch_size=1,
        raise_errors=True,
        cache_dir="/tmp",
        api_key="Not needed",
        force=False,
        progress_bar=False,
        save_results=False,
        device="cpu",  # We'll use CPU for this test
        verbose=True,
        trust_remote_code=False,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=False,
        num_iterations=1,
        api_base="http://vllm-service:8000/v1",
        api_version=None,
        debug=False,
        run_with_cli=False,
        only_allow_safetensors=False,
        use_internal_server=True,  # This flag enables the internal server module
    )
    
    # Model configuration for Llama-3.1-8B-Instruct
    model_config = ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="main",
        task="text-generation",
        languages=[EN],
        inference_backend=InferenceBackend.LITELLM,
        merge=False,
        model_type=ModelType.GENERATIVE,
        fresh=False,
        model_cache_dir="/tmp",
        adapter_base_model_id=None,
        internal_server=True,  # This also enables internal server
    )
    
    # Simple dataset configuration for text generation
    dataset_config = DatasetConfig(
        name="test_dataset",
        pretty_name="Test Dataset",
        huggingface_id="test_dataset",
        task=SUMM,  # Use the existing summarization task
        languages=[EN],  # English language
        max_generated_tokens=100,
        few_shot_examples=[],
        prompt_template="",
    )
    
    try:
        # Create the model instance
        print("Creating LiteLLMInternalModel instance...")
        model = LiteLLMInternalModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        print("✓ Model instance created successfully")
        
        # Test a simple generation
        print("\nTesting simple generation...")
        test_input = {
            "messages": [
                [
                    {"role": "user", "content": "Oslo is the "}
                ]
            ]
        }
        
        # Generate output
        output = model.generate(test_input)
        print("✓ Generation completed successfully")
        print(f"Generated text: {output.generated_texts[0]}")
        
        print("\n🎉 Internal vLLM module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing internal vLLM module with your inference server...")
    success = test_internal_vllm_server()
    sys.exit(0 if success else 1) 
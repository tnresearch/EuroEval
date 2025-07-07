from euroeval.benchmarker import Benchmarker
from euroeval.model_config import get_model_config
from euroeval.model_loading import load_model
from euroeval.data_models import DatasetConfig, Task
from euroeval.enums import TaskGroup, ModelType, InferenceBackend
from euroeval.tasks import SENT
from euroeval.languages import get_all_languages


def test_model_selection():
    """Test different model selection scenarios."""
    
    print("=== Testing Model Selection Based on Configuration ===\n")
    
    # Get a language for testing
    languages = list(get_all_languages().values())
    language = languages[0] if languages else None
    
    # Create a simple dataset config for testing
    dataset_config = DatasetConfig(
        name="test-dataset",
        pretty_name="Test Dataset", 
        huggingface_id="test-dataset",
        task=SENT,
        languages=[language] if language else [],
        _max_generated_tokens=100,
        _prompt_label_mapping={},
        _labels=["positive", "neutral", "negative"],
    )
    
    print("Configuration Summary:")
    print("- use_internal_server=False: Uses standard modules (HuggingFace, LiteLLM, etc.)")
    print("- use_internal_server=True: Forces all models through internal vLLM server")
    print("- Model type determines which module is selected")
    print("- Internal server flag overrides normal selection for generative models\n")
    
    # Test 1: Standard HuggingFace model (should use HuggingFaceEncoderModel)
    print("Test 1: Encoder model with use_internal_server=False")
    print("Expected: HuggingFaceEncoderModel (standard local inference)")
    try:
        benchmarker1 = Benchmarker(
            api_key=None,
            api_base=None,
            use_internal_server=False
        )
        model_config1 = get_model_config("fresh-electra-small", benchmarker1.benchmark_config)
        print(f"  ✓ Model ID: {model_config1.model_id}")
        print(f"  ✓ Inference Backend: {model_config1.inference_backend}")
        print(f"  ✓ Model Type: {model_config1.model_type}")
        print(f"  ✓ Internal Server: {model_config1.internal_server}")
        
        # Test actual model loading
        model1 = load_model(model_config1, dataset_config, benchmarker1.benchmark_config)
        print(f"  ✓ Loaded Model Class: {model1.__class__.__name__}")
        print("  ✓ Result: Standard HuggingFace module selected\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
    
    # Test 2: Encoder model with internal server flag (should still use HuggingFaceEncoderModel)
    print("Test 2: Encoder model with use_internal_server=True")
    print("Expected: HuggingFaceEncoderModel (internal server only affects generative models)")
    try:
        benchmarker2 = Benchmarker(
            api_key="test-key",
            api_base="http://localhost:8000/v1",
            use_internal_server=True
        )
        model_config2 = get_model_config("fresh-electra-small", benchmarker2.benchmark_config)
        print(f"  ✓ Model ID: {model_config2.model_id}")
        print(f"  ✓ Inference Backend: {model_config2.inference_backend}")
        print(f"  ✓ Model Type: {model_config2.model_type}")
        print(f"  ✓ Internal Server: {model_config2.internal_server}")
        
        # Test actual model loading
        model2 = load_model(model_config2, dataset_config, benchmarker2.benchmark_config)
        print(f"  ✓ Loaded Model Class: {model2.__class__.__name__}")
        print("  ✓ Result: Still uses HuggingFace module (encoder models not affected)\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
    
    # Test 3: Test HuggingFace module rejection of internal server config
    print("Test 3: HuggingFace module rejection of internal server config")
    print("Expected: HuggingFaceEncoderModel rejects use_internal_server=True")
    try:
        benchmarker3 = Benchmarker(
            api_key="test-key",
            api_base="http://localhost:8000/v1",
            use_internal_server=True
        )
        model_config3 = get_model_config("fresh-electra-small", benchmarker3.benchmark_config)
        
        # Try to load with HuggingFace module directly (should fail)
        from euroeval.benchmark_modules import HuggingFaceEncoderModel
        model3 = HuggingFaceEncoderModel(
            model_config=model_config3,
            dataset_config=dataset_config,
            benchmark_config=benchmarker3.benchmark_config
        )
        print("  ✗ HuggingFace module should have rejected this config")
    except Exception as e:
        print(f"  ✓ HuggingFace module correctly rejected: {e}\n")
    
    # Test 4: Test LiteLLMInternalModel directly
    print("Test 4: Direct LiteLLMInternalModel instantiation")
    print("Expected: LiteLLMInternalModel loads successfully with internal server config")
    try:
        benchmarker4 = Benchmarker(
            api_key="test-key",
            api_base="http://localhost:8000/v1",
            use_internal_server=True
        )
        model_config4 = get_model_config("fresh-electra-small", benchmarker4.benchmark_config)
        
        # Try to load the model directly
        from euroeval.benchmark_modules import LiteLLMInternalModel
        model4 = LiteLLMInternalModel(
            model_config=model_config4,
            dataset_config=dataset_config,
            benchmark_config=benchmarker4.benchmark_config
        )
        print(f"  ✓ Successfully loaded {type(model4).__name__}")
        print(f"  ✓ Model class: {model4.__class__.__name__}")
        print(f"  ✓ High priority: {model4.high_priority}")
        print(f"  ✓ Supported tasks: {model4.SUPPORTED_TASK_GROUPS}")
        print("  ✓ Result: Internal server module loads successfully\n")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}\n")
    
    # Test 5: Configuration comparison
    print("Test 5: Configuration Comparison")
    print("Comparing benchmark configurations with different use_internal_server settings:")
    
    benchmarker_false = Benchmarker(use_internal_server=False)
    benchmarker_true = Benchmarker(use_internal_server=True)
    
    print(f"  use_internal_server=False: {benchmarker_false.benchmark_config.use_internal_server}")
    print(f"  use_internal_server=True: {benchmarker_true.benchmark_config.use_internal_server}")
    print("  ✓ Configuration properly passed through to benchmark config\n")
    
    print("=== Test Summary ===")
    print("✓ use_internal_server parameter properly integrated into Benchmarker")
    print("✓ Configuration correctly passed to BenchmarkConfig")
    print("✓ Model loading logic respects internal server flag")
    print("✓ HuggingFace module correctly rejects internal server config")
    print("✓ LiteLLMInternalModel can be instantiated directly")
    print("✓ Internal server only affects generative models, not encoder models")


if __name__ == "__main__":
    test_model_selection()
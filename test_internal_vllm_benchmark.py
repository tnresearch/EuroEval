#!/usr/bin/env python3
"""Extended benchmark test for internal vLLM module using real EuroEval datasets."""

import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from euroeval.benchmark_modules.litellm_internal import LiteLLMInternalModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import ModelType, InferenceBackend, TaskGroup
from euroeval.tasks import SUMM, SENT, RC
from euroeval.languages import EN
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.data_loading import load_data
from numpy.random import default_rng


def test_internal_vllm_benchmark():
    """Test the internal vLLM module with real EuroEval datasets."""
    
    print("🚀 Starting comprehensive internal vLLM benchmark test...")
    
    # Configuration for your internal vLLM server
    benchmark_config = BenchmarkConfig(
        model_languages=[EN],
        dataset_languages=[EN],
        tasks=[SUMM, SENT, RC],  # Test multiple task types
        datasets=["cnn-dailymail", "sst5", "squad"],  # Real EuroEval datasets
        batch_size=1,
        raise_errors=True,
        cache_dir="/tmp",
        api_key="Not needed",
        force=False,
        progress_bar=True,
        save_results=False,
        device="cpu",
        verbose=True,
        trust_remote_code=False,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=False,  # Use zero-shot for faster testing
        num_iterations=1,
        api_base="http://vllm-service:8000/v1",
        api_version=None,
        debug=False,
        run_with_cli=False,
        only_allow_safetensors=False,
        use_internal_server=True,
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
        internal_server=True,
    )
    
    # Get real dataset configurations
    all_dataset_configs = get_all_dataset_configs()
    
    # Test datasets we want to benchmark
    test_datasets = ["cnn-dailymail", "sst5", "squad"]
    
    total_samples = 0
    total_time = 0
    
    for dataset_name in test_datasets:
        if dataset_name not in all_dataset_configs:
            print(f"⚠️  Dataset {dataset_name} not found, skipping...")
            continue
            
        dataset_config = all_dataset_configs[dataset_name]
        print(f"\n📊 Testing dataset: {dataset_config.pretty_name}")
        print(f"   Task: {dataset_config.task.name}")
        print(f"   Languages: {[lang.name for lang in dataset_config.languages]}")
        
        try:
            # Load the dataset
            print(f"   📥 Loading dataset...")
            datasets = load_data(
                rng=default_rng(seed=4242),
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )
            
            # Create the model
            print(f"   🤖 Creating model...")
            model = LiteLLMInternalModel(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )
            
            # Test on a few samples from each dataset
            test_samples = min(5, len(datasets[0]["test"]))
            print(f"   🧪 Testing on {test_samples} samples...")
            
            dataset_start_time = time.time()
            
            for i in range(test_samples):
                sample = datasets[0]["test"][i]
                
                # Prepare input based on task type
                if dataset_config.task.task_group == TaskGroup.TEXT_TO_TEXT:
                    # Summarization task
                    input_text = sample.get("text", sample.get("article", ""))
                    prompt = f"Document: {input_text}\n\nWrite a summary of the above document."
                    messages = [{"role": "user", "content": prompt}]
                    
                elif dataset_config.task.task_group == TaskGroup.SEQUENCE_CLASSIFICATION:
                    # Sentiment classification task
                    input_text = sample.get("text", "")
                    prompt = f"Text: {input_text}\n\nWhat is the sentiment of this text? Choose from: positive, neutral, negative."
                    messages = [{"role": "user", "content": prompt}]
                    
                elif dataset_config.task.task_group == TaskGroup.QUESTION_ANSWERING:
                    # Question answering task
                    context = sample.get("context", "")
                    question = sample.get("question", "")
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    messages = [{"role": "user", "content": prompt}]
                    
                else:
                    # Generic text generation
                    input_text = sample.get("text", "")
                    prompt = f"Complete the following text: {input_text}"
                    messages = [{"role": "user", "content": prompt}]
                
                # Generate output
                try:
                    output = model.generate({"messages": [messages]})
                    generated_text = output.generated_texts[0] if output.generated_texts else "No output"
                    
                    print(f"      Sample {i+1}:")
                    print(f"        Input: {prompt[:100]}...")
                    print(f"        Output: {generated_text[:100]}...")
                    
                    total_samples += 1
                    
                except Exception as e:
                    print(f"      ❌ Error on sample {i+1}: {e}")
            
            dataset_time = time.time() - dataset_start_time
            total_time += dataset_time
            print(f"   ⏱️  Dataset completed in {dataset_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error testing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Benchmark completed!")
    print(f"   Total samples tested: {total_samples}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per sample: {total_time/total_samples:.2f}s" if total_samples > 0 else "   No samples completed")
    
    return total_samples > 0


def test_single_dataset_benchmark():
    """Test a single dataset with more comprehensive evaluation."""
    
    print("\n🔬 Running detailed single dataset benchmark...")
    
    # Use CNN-DailyMail summarization dataset
    dataset_name = "cnn-dailymail"
    all_dataset_configs = get_all_dataset_configs()
    
    if dataset_name not in all_dataset_configs:
        print(f"❌ Dataset {dataset_name} not found")
        return False
    
    dataset_config = all_dataset_configs[dataset_name]
    
    # Configuration
    benchmark_config = BenchmarkConfig(
        model_languages=[EN],
        dataset_languages=[EN],
        tasks=[SUMM],
        datasets=[dataset_name],
        batch_size=1,
        raise_errors=True,
        cache_dir="/tmp",
        api_key="Not needed",
        force=False,
        progress_bar=True,
        save_results=False,
        device="cpu",
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
        use_internal_server=True,
    )
    
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
        internal_server=True,
    )
    
    try:
        # Load dataset
        print(f"📥 Loading {dataset_config.pretty_name}...")
        datasets = load_data(
            rng=default_rng(seed=4242),
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        
        # Create model
        print("🤖 Creating model...")
        model = LiteLLMInternalModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        
        # Test on more samples
        test_samples = min(10, len(datasets[0]["test"]))
        print(f"🧪 Testing on {test_samples} samples...")
        
        start_time = time.time()
        successful_generations = 0
        
        for i in range(test_samples):
            sample = datasets[0]["test"][i]
            article = sample.get("text", sample.get("article", ""))
            
            # Create summarization prompt
            prompt = f"Document: {article}\n\nWrite a summary of the above document."
            messages = [{"role": "user", "content": prompt}]
            
            try:
                output = model.generate({"messages": [messages]})
                generated_text = output.generated_texts[0] if output.generated_texts else "No output"
                
                print(f"\n📄 Sample {i+1}:")
                print(f"   Article (first 200 chars): {article[:200]}...")
                print(f"   Summary: {generated_text}")
                
                successful_generations += 1
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        total_time = time.time() - start_time
        print(f"\n📊 Results:")
        print(f"   Successful generations: {successful_generations}/{test_samples}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per generation: {total_time/test_samples:.2f}s")
        
        return successful_generations > 0
        
    except Exception as e:
        print(f"❌ Error in detailed benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing internal vLLM module with comprehensive benchmarks...")
    
    # Run basic multi-dataset benchmark
    success1 = test_internal_vllm_benchmark()
    
    # Run detailed single dataset benchmark
    success2 = test_single_dataset_benchmark()
    
    if success1 and success2:
        print("\n🎉 All benchmarks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some benchmarks failed!")
        sys.exit(1) 
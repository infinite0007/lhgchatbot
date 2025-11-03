import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available for PyTorch? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs PyTorch sees: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Current GPU Model (PyTorch): {torch.cuda.get_device_name(0)}")

print("\nAttempting to import Llama...")
try:
    from llama_cpp import Llama
    print("Llama imported successfully!")
    # For a more thorough test, you'd load a model with n_gpu_layers > 0
    llm = Llama(model_path="../gemma-3-4b-Instruct-GGUF/gemma-3-4b-it-Q2_K.gguf", n_gpu_layers=30) 
    print("Llama object initialized (this would test actual GPU offload).")
except Exception as e:
    print(f"Error importing or initializing Llama: {e}")

print("\nChecking CMAKE_ARGS from Python environment:")
print(f"CMAKE_ARGS: {os.environ.get('CMAKE_ARGS')}") 

quit()
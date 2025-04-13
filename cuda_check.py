import torch
import tensorflow as tf

print(f"pytorch version: {torch.__version__} \ntorch cuda: {torch.cuda.is_available()} \n ===============\n")
print(f"tensorflow version: {tf.__version__} \ntf cuda: {tf.test.is_gpu_available} \n ===============\ndevices: {tf.config.list_physical_devices()}")


def force_cuda_initialize():
    """Force CUDA initialization and verify it's working"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system!")
        return False

    print("\n==== GPU VERIFICATION ====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Force CUDA initialization by creating a small tensor
    test_tensor = torch.ones(10, 10)

    # Move to CUDA and check
    try:
        test_tensor = test_tensor.cuda()
        print(f"Test tensor device: {test_tensor.device}")

        # Try a simple operation
        result = test_tensor + test_tensor
        print(f"Test operation successful: {result.shape} on {result.device}")

        # Clear cache
        del test_tensor
        del result
        torch.cuda.empty_cache()
        print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print("✅ CUDA initialized successfully")
        print("==========================\n")
        return True
    except Exception as e:
        print(f"❌ CUDA initialization failed: {e}")
        print("==========================\n")
        return False

print(force_cuda_initialize())
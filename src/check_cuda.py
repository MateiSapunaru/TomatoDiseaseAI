import torch


def check_cuda():
    print("=" * 50)
    print("PyTorch CUDA Check")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        device_index = torch.cuda.current_device()

        print(f"Current device index: {device_index}")
        print(f"GPU name: {torch.cuda.get_device_name(device_index)}")

        tensor = torch.rand(1000, 1000).cuda()

        print("\nRunning small GPU benchmark...")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(100):
            tensor = tensor @ tensor

        end.record()

        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)

        print(f"GPU test completed successfully.")
        print(f"Execution time: {elapsed:.2f} ms")

    else:
        print("\nCUDA is NOT available.")
        print("Training will run on CPU.")

        print("\nPossible reasons:")
        print("1. NVIDIA driver not installed")
        print("2. CUDA version mismatch")
        print("3. PyTorch CPU-only version installed")
        print("4. GPU not detected")

    print("=" * 50)


if __name__ == "__main__":
    check_cuda()
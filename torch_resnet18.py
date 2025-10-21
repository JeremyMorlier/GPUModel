import torch
import torchvision.models as models
import torch.profiler

# Load ResNet18 model and set to evaluation mode
model = models.resnet18(pretrained=False).cuda().eval()
input_size = (1, 3, 224, 224)
input_tensor = torch.randn(input_size, dtype=torch.float32, device="cuda")

# Warm-up (optional, to avoid initial CUDA overhead)
for _ in range(3):
    _ = model(input_tensor)
torch.cuda.synchronize()

# Measure end-to-end latency
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
_ = model(input_tensor)
end_event.record()
end_event.synchronize()
latency_ms = start_event.elapsed_time(end_event)
print(f"End-to-End Latency: {latency_ms:.4f} ms")

# Profiling with torch.profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # Uncomment to save trace
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/resnet18"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    _ = model(input_tensor)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export trace (optional)
prof.export_chrome_trace("resnet18_trace.json")

# Verify output (optional)
print("Output shape:", _.shape)
print("Output dtype:", _.dtype)

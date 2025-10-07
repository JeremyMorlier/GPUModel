import torch
import torch.profiler

# Define matrix dimensions
D0, D1, D2 = 12800, 12800, 12800

# Create random FP16 tensors
input = torch.randn(D0, D2, dtype=torch.float16, device="cuda")
weight = torch.randn(D2, D1, dtype=torch.float16, device="cuda")
output = torch.zeros(D0, D1, dtype=torch.float16, device="cuda")

# Warm-up (optional, to avoid initial CUDA overhead)
for _ in range(3):
    output = torch.matmul(input, weight)

# Synchronize to ensure warm-up is complete
torch.cuda.synchronize()

# Measure latency
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = torch.matmul(input, weight)
end_event.record()
end_event.synchronize()

latency_ms = start_event.elapsed_time(end_event)
print(f"GEMM Latency: {latency_ms:.4f} ms")

# Profiling with torch.profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # schedule=torch.profiler.schedule(
    #     wait=1,
    #     warmup=1,
    #     active=3,
    # ),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/gemm_fp16"),
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True,
) as prof:
    for _ in range(5):
        output = torch.matmul(input, weight)

print(prof)
prof.export_chrome_trace("trace.json")
# Print profiling results (optional)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Verify output (optional)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)

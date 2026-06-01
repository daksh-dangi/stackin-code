import torch
torch.cuda.set_device(0)
import torch.nn.functional as F
import time
from hvp_triton import JVPAttn

def benchmark_double_backprop(
    B=4, H=32, L=2048, D=128, 
    dtype=torch.bfloat16, 
    iters=50, 
    warmup=10
):
    device = "cuda"
    
    # Initialize tensors
    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    
    # Tangents for the second backward pass (simulating grad_outputs of gradients)
    v_q = torch.randn_like(q)
    v_k = torch.randn_like(k)
    v_v = torch.randn_like(v)
    
    # Gradient output for the first backward
    grad_out = torch.randn(B, H, L, D, device=device, dtype=dtype)

    def run_step(mode="custom"):
        # Clear grads
        for p in [q, k, v]:
            p.grad = None
        
        # 1. Forward Pass
        if mode == "custom":
            out = JVPAttn.fwd(q, k, v, causal=True)
        else:
            # Modern PyTorch context manager
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH): 
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 2. First Backward (create_graph=True is the heavy part)
        grads = torch.autograd.grad(
            outputs=out,
            inputs=(q, k, v),
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True
        )
        
        # 3. Second Backward (HVP)
        hvp = torch.autograd.grad(
            outputs=grads,
            inputs=(q, k, v),
            grad_outputs=(v_q, v_k, v_v)
        )
        
        torch.cuda.synchronize()

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        run_step(mode="native")
        run_step(mode="custom")

    # Benchmark Loop
    results = {}
    for mode in ["native", "custom"]:
        print(f"Benchmarking {mode}...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        start_event.record()
        for _ in range(iters):
            run_step(mode)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / iters
        max_mem = torch.cuda.max_memory_allocated() / 1e9 # GB
        
        results[mode] = {"time": elapsed_time, "memory": max_mem}

    # Final Report
    print("\n--- Benchmark Results (Double Backprop Step) ---")
    print(f"Config: Batch={B}, Heads={H}, Seq={L}, Dim={D}, Dtype={dtype}")
    for mode, metrics in results.items():
        print(f"[{mode.upper()}] Avg Time: {metrics['time']:.2f} ms | Max Memory: {metrics['memory']:.2f} GB")
    
    speedup = results["native"]["time"] / results["custom"]["time"]
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_double_backprop()
# GradMem

## Initial Steps
This project is a work in progress. I have updated and validated the kernel code that was originally provided by the authors of [GradMem](https://arxiv.org/pdf/2603.13875v1) located here: https://github.com/yurakuratov/gradmem/blob/main/attn_double_bwd/hvp_semi_manual.py . The custom kernel currently operates at a **~55x memory reduction** compared to the baseline. However, I want to be clear that this kernel was built entirely on top of their initial effort. I do not want to make any claims of novelty.

I extended it beyond their initial manual implementation *solely through the use of AI*. I had never worked at the kernel level before, and GradMem's algorithm required materializing double-backward (Hessian-Vector product) graphs, unrolled across multiple inner loop optimization steps. To bypass the enormous memory footprint of PyTorch's native `create_graph=True` pipeline, I used Claude to help finish a fused Flash-Attention HVP Triton kernel that is far more compute friendly. It only materializes the graph for the initial backwards pass, and uses the custom HVP for the second pass.

Following this, I extended it further towards a Forward-over-Reverse (FoR) implementation; I found this necessary as a quick profiling run of the kernel showed that 49% of the total CUDA time was spent in `_attn_double_bwd_q_kernel`, and another 40% in `_attn_double_bwd_kv_kernel`. To mitigate this, and the graph materialization *entirely*, I pointed Claude to [this blog post](https://iclr-blogposts.github.io/2024/blog/bench-hvp/) containing the implementation details for the FoR kernels. 

This implementation yielded even more drastic improvement, but came with its own set of issues. I have set it aside for the time being to opt for progress in a direction where I have more control - actually implementing GradMem.

## Benchmarks
The figures shown below are obtained from running the tests on an **H200 SXM** with shapes `B=1, H=32, S=8192, head_dim=128, bf16`, on a full WRITE loop (the inner loop described in GradMem) comparing against PyTorch's eager `create_graph=True` double-backward pass. To be clear, the actual training runs I am testing with are on sequence lengths far smaller than 8192 (validating my approach and implementation) - so the memory savings are proportionally less, but will become a lot more noticeable once I begin to scale up.

### FA vs. Eager PyTorch
| K (inner steps) | Kernel (ms) | Eager (ms) | Speedup | Kernel VRAM | Eager VRAM | Memory ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 158.0 | 70.1 | 0.44x | 836.00 MiB | 44.25 GiB | ~0.02x |
| 2 | 314.4 | 137.8 | 0.44x | 965.00 MiB | 52.25 GiB | ~0.02x |
| 3 | 466.6 | 207.0 | 0.44x | 1.07 GiB | 60.25 GiB | ~0.02x |
| 4 | 621.8 | 274.4 | 0.44x | 1.19 GiB | 68.25 GiB | ~0.02x |

Here, we can see a strong reduction of **~55x**. However, it is quite compute inefficient - displaying a **slowdown of 2.26x**. Ideally, before doing a complete run, I would integrate the FoR kernel into my training pipeline, shown below.

### Forward-over-Reverse vs. Eager PyTorch
| K (inner steps) | Kernel (ms) | Eager (ms) | Speedup | Kernel VRAM | Eager VRAM | Memory ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 119.9 | 69.3  | 0.58× | 836 MiB | 44.25 GiB | ~0.02× |
| 2 | 176.5 | 137.1 | 0.78× | 836 MiB | 52.25 GiB | ~0.02× |
| 3 | 234.0 | 204.6 | 0.87× | 836 MiB | 60.25 GiB | ~0.01× |
| 4 | 290.3 | 271.4 | 0.93× | 836 MiB | 68.25 GiB | ~0.01× |

This paints an even rosier picture. We can see that for the FoR kernel, the peak memory stays **constant at 836 MiB across all K** because the graphs are never materialized, whereas the eager implementation grows linearly with the number of unrolled inner steps (K) - reaching 68.25 GiB at K=4. This is an **~80x memory reduction**. The kernel reaches performance parity with the eager implementation as K grows, evidenced by the 0.93x speedup (or 1.07x slowdown) at K=4. It trades a small amount of compute for the memory headroom that would otherwise OOM for larger K / sequence lengths.

## Status & Caveats
Currently, I have optimized this for, and validated it on, **Qwen3-4B** only. Other architectures would fall back to the regular eager implementation. I am currently in the process of stabilizing training runs for smaller models, figuring out the best hyperparameters that promote a stable run; and actually encourage compression into the Memory Tokens. 

The paper itself had opted for training on a synthetic task, and **not** a Natural Language one (like QAs or needle-in-a-haystack style questions), and the results from the initial faithful reconstruction of the paper highlighted that disparity. I am working towards re-defining the outer loop(READ)'s objective to actually encourage compression; as my initial tests have shown the memory tokens are undergoing representational collapse.

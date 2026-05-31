## Write-up

The method and results are documented as a four-part series:

1. **[Dr. GRPO with Gradient Regularization](https://daksh-dangi.github.io/stacking-paper/articles/grad-reg)** - method setup, motivation, and the science-domain datasets
2. **[Training Qwen-4B](https://daksh-dangi.github.io/stacking-paper/articles/training-grad-reg)** - the training pipeline, debugging (vanishing gradients, the vLLM logit-scaling bug), and the in-domain result
3. **[Mechanistic Analysis](https://daksh-dangi.github.io/stacking-paper/articles/evaluating-grad-reg)** - localizing learning via weight-delta norms, the detokenization hypothesis, and the entropy-collapse finding
4. **[Depth-Scaled GradReg](https://daksh-dangi.github.io/stacking-paper/articles/grad-reg-decay)** - turning the mechanistic finding into a depth-scaled design, plus an adversarial-probe analysis of the robustness/hallucination tradeoff

## Models

| Model | Description | Link |
|---|---|---|
| `gradReg` | Dr.GRPO + explicit gradient regularization | https://huggingface.co/daktshh/gradReg |
| `gradReg_decay` | Depth-scaled regularization + early-layer directional noise | https://huggingface.co/daktshh/gradReg_decay |

## Results

Mean ± std over 3 trials. GPQA uses greedy decoding (temp 0.0), MMLU-Pro uses sampling (temp 0.6)

| Model | GPQA Diamond (in-domain) | History MMLU-Pro (OOD) | Math MMLU-Pro (OOD) |
|---|---|---|---|
| **gradReg_decay** | **67.85 ± 1.62** | **47.68 ± 1.52** | **36.79 ± 0.20** |
| gradReg | 65.32 ± 1.05 | 47.24 ± 0.95 | 36.24 ± 0.73 |
| base | 64.31 ± 2.04 | 46.98 ± 1.14 | 36.27 ± 0.56 |
| grpo | 64.14 ± 2.02 | 44.97 ± 1.97 | 33.65 ± 0.42 |

## Training & evaluation

- **Dataset:** CamelAI 60k biology + chemistry + physics (open-ended, non-MCQ)
- **Base model:** Qwen3-4B
- **Compute:** trained on 2× H200-SXM
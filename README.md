# V-STaR: Reimplementation of "Training Verifiers for Self-Taught Reasoners"

This repository contains a reimplementation of the method proposed in the paper:

**V-STaR: Training Verifiers for Self-Taught Reasoners**  
Hosseini, Arian, et al.  
[arXiv preprint arXiv:2402.06457 (2024)](https://arxiv.org/pdf/2402.06457)

---

## Abstract

Common self-improvement approaches for large language models (LLMs), such as STaR, iteratively fine-tune LLMs on self-generated solutions to improve their problem-solving ability. However, these approaches discard the large amounts of incorrect solutions generated during this process, potentially neglecting valuable information in such solutions. To address this shortcoming, we propose **V-STaR**, which utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using **DPO** that judges the correctness of model-generated solutions. This verifier is used at inference time to select one solution among many candidate solutions. Running V-STaR for multiple iterations results in progressively better reasoners and verifiers, delivering a 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on common code generation and math reasoning benchmarks with LLaMA2 models.

---

## Implementation Details

This repository is a reimplementation of the V-STaR methodology using PyTorch and Hugging Face's Transformer library. It reproduces key results and experiments from the paper, including:
- Training a verifier using both correct and incorrect solutions.
- Iterative fine-tuning of LLMs for self-improvement.
- Evaluation on benchmarks for code generation and mathematical reasoning.

---

## Citation

If you use this reimplementation in your work, please cite the original paper:

```bibtex
@article{hosseini2024v,
  title={V-star: Training verifiers for self-taught reasoners},
  author={Hosseini, Arian and Yuan, Xingdi and Malkin, Nikolay and Courville, Aaron and Sordoni, Alessandro and Agarwal, Rishabh},
  journal={arXiv preprint arXiv:2402.06457},
  year={2024}
}

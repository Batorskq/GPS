<div align="center">
  <h1>GPS: General Per-Sample Prompter</h1>

  Paweł Batorski, Paul Swoboda

  [![arXiv](https://img.shields.io/badge/arXiv-2511.21714-red)](https://arxiv.org/abs/2511.21714)

</div>

<div align="center">
  <img src="imgs/piast.png" alt="PIAST overview figure" width="850">
</div>

---

This repository contains the official implementation of the paper  
**[GPS: General Per-Sample Prompter](https://arxiv.org/abs/2511.21714)**

## Abstract

LLMs are sensitive to prompting, with task performance often hinging on subtle, sometimes imperceptible variations in phrasing. As a result, crafting effective prompts manually remains challenging and time-consuming. Recent automatic prompting methods mitigate this difficulty but face three key limitations: (i) for each new task, they require large datasets to train good prompts;(ii) they rely on costly optimization loops that may take hours; (iii)they typically produce a single task-level prompt that does not adapt to the individual input problem to be solved.
We propose GPS, the first general-purpose, per-sample prompting method. Without any task-specific tuning, GPS generates a tailored prompt for each unseen input, improving performance across diverse tasks. The prompter is trained with reinforcement learning on a suite of training tasks and includes a novel regularization for effectively adapting to per-sample prompting. Finally, we employ Minimum Bayes Risk decoding to stabilize inference.
Empirically, GPS demonstrates competitive performance: we attain second best results among baselines on text simplification, third best results on summarization and on-par results on classification, while not training on any of these tasks, in contrast to the baselines. For in-domain prompting, we obtain sota on GSM8K. Our work shows the potential of a novel and effective paradigm for automatic prompting: generating adaptive, input-specific prompts without extensive optimization and without access to a task-specific training set. 

---

## ✨ Highlights

- 🌍 General-purpose, zero-shot prompting: generates strong prompts for unseen tasks without any task-specific dataset or tuning

- 🧩 Per-sample adaptation: produces a unique prompt per input (not one prompt per task), improving robustness on diverse examples

- 🛡️ Leakage-resistant training: includes LLM-judge and sample-swap regularization to prevent embedding answers inside prompts

- 🎲 Stable inference with MBR: uses Minimum Bayes Risk decoding over multiple sampled prompts (majority-vote for classification, ROUGE-consensus for generation)
---

## 🚀 Quickstart

### Sampling Regularization

```bash
./sampling_regularization.sh
```

### LLM as a Judge Regularization

```bash
./llm_regularization.sh
```
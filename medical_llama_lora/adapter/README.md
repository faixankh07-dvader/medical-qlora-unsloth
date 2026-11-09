---
base_model: unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: llama3.2
language:
- en
datasets:
- medalpaca/medical_meadow_medical_flashcards
tags:
- base_model:adapter:unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit
- lora
- qlora
- sft
- transformers
- trl
- unsloth
- medical
---

# Medical QLoRA adapter for Llama 3.2 3B Instruct

A LoRA adapter trained with QLoRA (4-bit) on medical Q&A flashcards. It makes Llama 3.2 3B Instruct answer medical questions in a short, flashcard-like style.

> **Not for clinical use.** Educational project. Outputs are not medical advice and can be wrong.

## Model details

- **Developed by:** M Faizan Khan
- **Base model:** `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`
- **Adapter type:** LoRA (r=16, alpha=16, dropout 0) on q, k, v, o, gate, up and down projections; 24.3M trainable parameters (0.75%)
- **Language:** English
- **License:** Llama 3.2 Community License (inherited from the base model)

## How to use

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path-or-repo-id-of-this-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Use the system prompt from training: *"You are a helpful medical assistant. Answer clinical questions accurately and concisely."*

## Training details

- **Data:** 5,000 randomly sampled (seed 3407) examples from `medalpaca/medical_meadow_medical_flashcards` (33,955 total), formatted with the Llama 3.1 chat template; loss computed on assistant answers only.
- **Procedure:** TRL `SFTTrainer`, 1 epoch (625 steps), effective batch size 8, learning rate 2e-4 with linear schedule and 10 warm-up steps, AdamW 8-bit, weight decay 0.01, fp16, max sequence length 2048.
- **Hardware:** Google Colab Tesla T4. Training took 20.5 minutes; peak GPU memory 3.66 GB.
- **Final average training loss:** 0.7479.

## Evaluation

Only a qualitative before/after check on a handful of medical questions; no benchmark or held-out validation set was used. The fine-tuned model's answers are shorter and more flashcard-like but not measurably more accurate, and several contain factual errors (for example, stating that metformin inhibits AMPK).

## Bias, risks and limitations

Trained on a small slice of short, wiki-derived flashcards for one epoch. The model can hallucinate, oversimplify, or give outdated or incorrect clinical guidance. Do not use it for diagnosis, treatment or any real medical decision.

### Framework versions

- PEFT 0.21.2
- Unsloth 2026.10.9, Transformers 5.17.0, PyTorch 2.11.0

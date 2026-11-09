# Medical QLoRA Fine-Tuning with Unsloth

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faixankh07-dvader/medical-qlora-unsloth/blob/main/notebooks/Medical_QLoRA_Unsloth.ipynb)

QLoRA (4-bit quantization + LoRA) fine-tuning of **Llama 3.2 3B Instruct** on medical Q&A flashcards, using [Unsloth](https://github.com/unslothai/unsloth) on a free Google Colab **Tesla T4**.

> **Disclaimer:** Educational project only. The model's outputs are **not medical advice** and must not be used for clinical decisions.

## Highlights

- Full workflow: 4-bit model loading → chat-template tokenization → LoRA adapter setup → epoch-based SFT → memory monitoring → adapter saving → reload and test.
- Trains **0.75%** of the model's parameters (24.3M of 3.24B).
- Fine-tuning peaked at **3.66 GB** of GPU memory (25% of the T4's 14.56 GB) and took **20.5 minutes**.
- The saved LoRA adapter is **97 MB**.

## Setup

| Item | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct` (4-bit NF4) |
| Dataset | [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards), 5,000 of 33,955 examples (seed 3407) |
| Method | QLoRA + supervised fine-tuning (TRL `SFTTrainer`), loss on assistant answers only |
| LoRA | r=16, alpha=16, dropout 0, all 7 projection modules (q, k, v, o, gate, up, down) |
| Training | 1 epoch (625 steps), effective batch 8, lr 2e-4 (linear, 10 warm-up steps), AdamW 8-bit, fp16 |
| Hardware | Colab free tier, Tesla T4 |
| Versions | Unsloth 2026.10.9, Transformers 5.17.0, PyTorch 2.11.0 |

## Results

| Metric | Value |
|---|---|
| Training time | 1,227 s (20.5 min), 4.07 samples/s |
| Final (average) training loss | 0.7479 |
| Peak GPU memory | 3.656 GB (25.1% of 14.56 GB) |
| Extra memory used by LoRA training | 1.23 GB |
| Adapter size | 97.3 MB |

![Training loss](images/loss_curve.png)

### What the fine-tune did (and didn't) do

Comparing answers before and after training on medical questions shows the model became **shorter and more flashcard-like**, matching the training data's style. It did **not** measurably improve medical accuracy, and some answers got worse:

- **Metformin:** the fine-tuned model said it *inhibits* AMPK (it activates it). The base model was correct.
- **Pneumonia:** it named macrolides as first-line; amoxicillin or doxycycline are typical outpatient first-line choices.
- **Hypertension:** it dropped thiazide diuretics and listed beta-blockers, which are not first-line.

With one epoch on 5,000 short flashcards and no held-out evaluation, this is expected: the data mostly teaches response format, and the base model already knew most of this material. See the [project report](report/Medical_QLoRA_Report.pdf) for the full before/after table.

## Usage

Open the notebook in Colab (T4 runtime) and run all cells, or load the saved adapter directly:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="adapter",        # folder containing the adapter files (or the Hugging Face repo id)
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant. Answer clinical questions accurately and concisely."},
    {"role": "user", "content": "What are the side effects of ACE inhibitors?"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

The adapter was trained with the system prompt shown above, so include it at inference.

## Repository structure

```
├── notebooks/Medical_QLoRA_Unsloth.ipynb   # full training notebook with saved outputs
├── adapter/                                 # LoRA adapter, config and model card
├── report/Medical_QLoRA_Report.pdf          # project report
└── images/                                  # figures and screenshots
```

## Limitations

- Only 5,000 of 33,955 examples and a single epoch.
- No validation split and no quantitative accuracy benchmark; loss does not measure factual correctness.
- The data is short flashcard-style facts, not realistic clinical cases.
- The model can hallucinate or oversimplify.

## Future work

Hold out a validation split, train longer on the full dataset, evaluate on a medical benchmark (e.g. MedMCQA, PubMedQA), and try a larger or reasoning-oriented base such as DeepSeek-R1-Distill-Llama-8B.

## Acknowledgements and license

Built with [Unsloth](https://github.com/unslothai/unsloth), TRL and PEFT. Data: [MedAlpaca](https://huggingface.co/medalpaca). The adapter is derived from Llama 3.2 and is subject to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).

Author: M Faizan Khan ([@faixankh07-dvader](https://github.com/faixankh07-dvader))

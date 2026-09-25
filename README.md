# Mini-LLM — GPT-Style Language Model From Scratch

A small decoder-only Transformer language model built from scratch with PyTorch.

This project explores the complete language-model development pipeline: dataset preparation, BPE tokenization, causal self-attention, pretraining, text generation, instruction fine-tuning, and answer-only supervised fine-tuning.

The goal is not to compete with production LLMs, but to understand and implement the core components behind modern autoregressive language models.

## Architecture

| Component | Configuration |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~13M |
| Vocabulary | 16,000 tokens |
| Context length | 256 tokens |
| Embedding dimension | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Attention | Causal multi-head self-attention |
| Activation | GELU |
| Optimizer | AdamW |
| Framework | PyTorch |
| Training device | Apple Silicon (MPS) |

The model follows the standard autoregressive language-modeling objective:

```text
tokens
  ↓
token embeddings + positional embeddings
  ↓
Transformer Block × 6
  ├── LayerNorm
  ├── Causal Multi-Head Self-Attention
  ├── Residual Connection
  ├── LayerNorm
  ├── MLP (4× expansion + GELU)
  └── Residual Connection
  ↓
Final LayerNorm
  ↓
Linear Language Modeling Head
  ↓
Next-token probabilities
```

## Data Pipeline

Pretraining data is prepared from a streamed subset of FineWeb-Edu.

The preprocessing pipeline includes:

```text
Raw documents
      ↓
Cleaning
      ↓
Exact deduplication
      ↓
BPE tokenizer training
      ↓
Tokenization
      ↓
Train / validation split
      ↓
Binary token files
      ↓
PyTorch training batches
```

The custom Byte-Pair Encoding tokenizer has a vocabulary size of 16,000 tokens.

The pretraining corpus used in the experiment contained approximately:

- 11.1M total tokens
- 10.0M training tokens
- 1.1M validation tokens

## Pretraining

The model was initialized from random weights and trained using next-token prediction.

Each input sequence:

```text
[x1, x2, x3, ..., xn]
```

is trained against the shifted target:

```text
[x2, x3, x4, ..., xn+1]
```

Cross-entropy loss is used to optimize the model.

During training, validation loss decreased from approximately:

```text
9.74 → 4.69
```

over 30,000 training steps.

This transformed the randomly initialized network from producing essentially random tokens into a model capable of generating English-like text.

## Instruction Fine-Tuning

After pretraining, the base model was instruction-tuned using question-answer examples derived from the Databricks Dolly dataset.

Two approaches were implemented.

### SFT v1 — Full Sequence Loss

The first approach calculated language-model loss across the complete sequence:

```text
Context + Question + Answer
```

This improved the model's exposure to question-answer formatting but still trained it to predict prompt tokens.

### SFT v2 — Answer-Only Loss

The second implementation masks prompt tokens from the loss calculation:

```text
Question: What is machine learning?
Answer: Machine learning is ...

Targets:

Question tokens → -100 (ignored)
Answer tokens   → supervised targets
```

PyTorch's `cross_entropy(..., ignore_index=-100)` ensures that optimization is performed only on answer tokens.

This more closely resembles supervised instruction fine-tuning used for instruction-following language models.

Validation answer-loss decreased from approximately:

```text
5.08 → 4.07
```

during the SFT experiment.

## Generation

Autoregressive generation is implemented manually.

For every generated token:

1. The latest context window is passed through the Transformer.
2. The logits of the final position are selected.
3. Temperature scaling is applied.
4. Top-k filtering restricts sampling to likely candidates.
5. A token is sampled from the resulting probability distribution.
6. Generation stops when the EOS token is produced.

Example:

```text
Question:
What is the capital of France?

Model output:
The capital of Germany is the capital of Britain, the capital of Spain,
Poland, Italy, France, Germany
```

The answer is factually incorrect, but the experiment demonstrates an important limitation: supervised fine-tuning can teach response structure and task behavior, while factual knowledge and language quality remain constrained by the capacity and pretraining of the base model.

## Project Structure

```text
mini-llm/
│
├── configs/
│   └── model.yaml
│
├── data/
│   └── tokenizer/
│       └── tokenizer.json
│
├── notebooks/
│   └── explore_dataset.ipynb
│
├── src/
│   ├── data/
│   │   ├── clean.py
│   │   ├── download.py
│   │   ├── prepare_tokens.py
│   │   ├── prepare_qa.py
│   │   ├── prepare_qa_tokens.py
│   │   ├── prepare_qa_sft.py
│   │   ├── profile_dataset.py
│   │   └── tokenize_dataset.py
│   │
│   ├── model/
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── gpt.py
│   │
│   ├── train.py
│   ├── generate.py
│   ├── finetune.py
│   ├── finetune_sft.py
│   ├── chat.py
│   └── chat_sft.py
│
├── requirements.txt
└── README.md
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the pretraining dataset:

```bash
python src/data/download.py
python src/data/clean.py
python src/data/tokenize_dataset.py
python src/data/prepare_tokens.py
```

Train the base model:

```bash
python src/train.py
```

Prepare instruction-tuning data:

```bash
python src/data/prepare_qa_sft.py
```

Run answer-only supervised fine-tuning:

```bash
python src/finetune_sft.py
```

Chat with the fine-tuned model:

```bash
python src/chat_sft.py
```

## What I Learned

This project was built to explore the mechanics of language models rather than relying on high-level training abstractions.

It includes hands-on implementations of:

- Byte-Pair Encoding tokenization
- causal self-attention
- multi-head attention
- Transformer residual blocks
- next-token prediction
- autoregressive generation
- checkpointing and validation
- supervised instruction fine-tuning
- prompt masking and answer-only loss
- temperature and top-k sampling
- Apple Silicon MPS training

One of the main observations from the experiments is that instruction fine-tuning and pretraining solve different problems. Fine-tuning can strongly influence how a model responds, but it cannot compensate for insufficient base-model capacity or limited pretraining knowledge.

## Future Work

Planned experiments include:

- validation-based best-checkpoint saving
- improved generation and repetition control
- larger scratch-trained models
- LoRA / QLoRA fine-tuning
- comparison with a pretrained small language model
- retrieval-augmented generation (RAG)
- evaluation across a fixed benchmark question set

## Disclaimer

This is an educational/research implementation. The model is intentionally small and its generated responses should not be treated as reliable factual information.
# Shakespeare Text Generator — Character-Level GPT (PyTorch)

A GPT-style **decoder-only Transformer** trained at the **character level** on the Shakespeare corpus to generate Shakespeare-like text from a seed prompt.


This repository contains:
- **`shakespear.py`**: end-to-end training + validation + checkpointing + loss plot + text generation
- **`generate.py`**: load a saved checkpoint (`ckpt.pt`) and generate text
- **`input.txt`**: Shakespeare dataset

## What this project does

- Builds a character vocabulary from `input.txt`
- Trains an autoregressive Transformer to predict the **next character**
- Uses **causal self-attention** (masking future tokens)
- Saves a checkpoint (`ckpt.pt`) during training
- Generates text from a prompt (default: `"O God, O God!"`)

## Model architecture (decoder-only Transformer)

Implemented in PyTorch with:
- token embeddings + positional embeddings
- stacked Transformer blocks (pre-layernorm)
  - causal multi-head self-attention
  - MLP / feed-forward network
  - residual connections + dropout
- final LayerNorm + final linear projection to vocabulary logits

## Default hyperparameters

Defined at the top of the scripts:
- `block_size = 128` (context length)
- `batch_size = 128`
- `nb_layers = 12`
- `nb_heads = 8`
- `nb_embd = 768`
- `lr = 5e-4`
- `train_steps = 6000`
- `residual_pdrop = 0.1`
- `embd_pdrop = 0.2`

## Results (from experiments)

Example generated text (seed: `"O God, O God!"`):
> O God, O God! that e'er this tongue of mine,  
> That laid the sentence of dread banishment  
> On yon proud man, should take it off again  
> With words of sooth! ...


- training loss converged to ~**0.155**
- validation loss around **3.26** (after 6000 steps with LR decay + gradient clipping + increased embedding dropout)
![Training vs Validation Loss](Losses.png)


## Requirements

- Python 3.9+ recommended
- PyTorch
- matplotlib



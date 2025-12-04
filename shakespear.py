import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

residual_pdrop = 0.1  # Attention is All You Need Paper
attention_pdrop = 0.1
block_size = 128
batch_size = 128
nb_layers = 12
nb_heads = 8
nb_embd = 768
lr = 3e-4
train_steps = 6000


class CharDataset(Dataset):
    """Emits batches of characters

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, data, block_size):

        self.block_size = block_size

        chars = sorted(list(set(data)))  # get characters from the input data

        self.stoi = {
            ch: i for i, ch in enumerate(chars)
        }  # mapping character to integer indeces
        self.itos = {
            i: ch for i, ch in enumerate(chars)
        }  # mapping integer indices to characters

        vocab_size = len(chars)
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return (
            len(self.data) - self.block_size
        )  # number of possible starting positions for a chunk of size block_size + 1

    def __getitem__(self, idx):
        chunk = self.data[
            idx : idx + self.block_size + 1
        ]  # grab a chunk of block_size + 1 characters from the data

        chunk_dict = [
            self.stoi[ch] for ch in chunk
        ]  # encode every character to an integer

        input_t = torch.tensor(chunk_dict[:-1])
        target_t = torch.tensor(chunk_dict[1:])

        return input_t, target_t  # return the chunk and the shifted version as tensors


text = open("input.txt", "r").read()
n = int(0.9 * len(text))
train_text = text[0:n]
val_text = text[n:]

train_ds = CharDataset(train_text, block_size)
val_ds = CharDataset(val_text, block_size)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True
)  # I get batches of (B,N) : B=batch_size examples(sequences) of N = block_size each
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, nb_embd, nb_heads, block_size, attention_pdrop, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.nb_heads = nb_heads
        self.attendion_pdrop = attention_pdrop
        self.residual_pdrop = residual_pdrop
        self.block_size = block_size

        # nb_embd % nb_heads == 0
        self.attention = nn.Linear(
            nb_embd, 3 * nb_embd
        )  # I get for each token in input a vector of size 3*nb_embd
        # A big Weight matrix of shape [3*nb_embd,nb_embd] that can be considered
        # the concatenation (by last dimension) of Wq,Wk,Wv

        # I create the mask for causal self attention
        # with a lower triangular matrix so that no information from future flows to the past
        # in each position i, only elements in positions <= i are not zero
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

    def forward(self, x):
        B, N, C = x.size()
        h = self.nb_heads

        qkv = self.attention(x)
        q, k, v = qkv.chunk(
            3, dim=-1
        )  # I split the big matrix, into 3 separate Q,K,V matrices of shape [N,C]
        # The q,k,v matrices are calculated for all heads in batch at once , shape [B,N,C]
        q = q.view(B, N, h, C // h).transpose(
            1, 2
        )  # I first split the channel dimension (nb_embd=C) into nb_heads * head_dimension
        # shape [B,heads,N(sequence),head_dimension]
        k = k.view(B, N, h, C // h).transpose(1, 2)
        v = v.view(B, N, h, C // h).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # I apply the mask for causal self-attention
        # no information from the future flows to the past
        # I use a lower triangular matrix to do this
        mask = self.mask[:, :, N, N]
        att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

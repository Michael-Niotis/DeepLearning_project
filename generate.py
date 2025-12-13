import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


residual_pdrop = 0.1  # Attention is All You Need Paper
embd_pdrop = 0.2
block_size = 128
batch_size = 128
nb_layers = 12
nb_heads = 8
nb_embd = 768
lr = 5e-4
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

        input_t = torch.tensor(chunk_dict[:-1], dtype=torch.long)
        target_t = torch.tensor(chunk_dict[1:], dtype=torch.long)

        return input_t, target_t  # return the chunk and the shifted version as tensors


class CausalSelfAttention(nn.Module):
    def __init__(self, nb_embd, nb_heads, block_size, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.nb_heads = nb_heads

        self.residual_pdrop = residual_pdrop
        self.block_size = block_size

        # nb_embd % nb_heads == 0
        self.attention = nn.Linear(
            self.nb_embd, 3 * self.nb_embd
        )  # I get for each token in input a vector of size 3*nb_embd
        # A big Weight matrix of shape [3*nb_embd,nb_embd] that can be considered
        # the concatenation (by last dimension) of Wq,Wk,Wv

        self.out_proj = nn.Linear(
            self.nb_embd, self.nb_embd
        )  # the output projection W0 according to paper

        # Apply dropout to the output according to "Attention is all you need" before normalization and residual connection
        self.residual_dropout = nn.Dropout(self.residual_pdrop)

        # I create the mask for causal self attention
        # with a lower triangular matrix so that no information from future flows to the past
        # in each position of the sequence i, only elements in positions <= i are not zero
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
        )  # I split the big matrix, into 3 separate Q,K,V matrices of shape [B,N,C]
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
        mask = self.mask[:, :, :N, :N]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B,heads,N,N) x (B,heads,N, head_dim) --> (B,heads,N,head_dim)
        out = (
            out.transpose(1, 2).contiguous().view(B, N, C)
        )  # merge the last 2 dimensions (heads,head_dim) into one (C = heads * head_dim) to perform the projection W0

        # perform the final projection to the concatenation of all heads
        out = self.out_proj(out)
        # perform a dropout to the output of this sub-layer according to the paper
        out = self.residual_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, nb_embd, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.residual_pdrop = residual_pdrop
        self.mlp = nn.Sequential(
            nn.Linear(self.nb_embd, 4 * self.nb_embd),
            nn.ReLU(),
            nn.Linear(4 * self.nb_embd, self.nb_embd),
        )

        self.residual_dropout = nn.Dropout(self.residual_pdrop)

    def forward(self, x):
        ffn_out = self.mlp(x)
        ffn_out = self.residual_dropout(ffn_out)
        return ffn_out


class Block(nn.Module):
    def __init__(self, nb_embd, nb_heads, block_size, residual_pdrop):
        super().__init__()
        self.nb_embd = nb_embd
        self.nb_heads = nb_heads
        self.block_size = block_size
        self.residual_pdrop = residual_pdrop

        self.CausalSelfAttention = CausalSelfAttention(
            self.nb_embd, self.nb_heads, self.block_size, self.residual_pdrop
        )
        self.mlp = MLP(self.nb_embd, self.residual_pdrop)
        self.LayerNorm_1 = nn.LayerNorm(self.nb_embd)
        self.LayerNorm_2 = nn.LayerNorm(self.nb_embd)

    def forward(self, x):
        x = x + self.CausalSelfAttention(self.LayerNorm_1(x))
        x = x + self.mlp(self.LayerNorm_2(x))
        return x


class FullModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        nb_embd,
        block_size,
        nb_layers,
        nb_heads,
        embd_pdrop,
        residual_pdrop,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.nb_embd = nb_embd
        self.block_size = block_size
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.residual_pdrop = residual_pdrop
        self.embd_pdrop = embd_pdrop
        self.wte = nn.Embedding(
            self.vocab_size, self.nb_embd
        )  # I create a lookup_table for the token ID's
        self.pte = nn.Embedding(
            self.block_size, self.nb_embd
        )  # I create a lookup_table for the positions
        self.dropout = nn.Dropout(
            self.embd_pdrop
        )  # dropout to be applied in the input (sum of tok_emb and pos_emb)
        self.blocks = nn.ModuleList(
            [
                Block(self.nb_embd, self.nb_heads, self.block_size, self.residual_pdrop)
                for _ in range(self.nb_layers)
            ]
        )

        self.Final_LayerNorm = nn.LayerNorm(self.nb_embd)
        self.lm_head = nn.Linear(self.nb_embd, self.vocab_size)

    def forward(self, idx):
        B, N = idx.size()  # sequences of length N
        assert N <= self.block_size
        pos = torch.arange(0, N, device=idx.device).unsqueeze(
            0
        )  # transform into a row vector(tensor) of shape (1,N) so I can later broadcast it
        tok_emb = self.wte(idx)  # create the token embeddings, shape of B,N,nb_embd
        pos_emb = self.pte(pos)  # positional embeddings, shape of 1,N,nb_embd
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.Final_LayerNorm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )  # Take the whole sequence if it's smaller than block_size, otherwise take the last block_size elements

            logits = self(idx_cond)  # [B,N,vocab_size]

            logits = logits[:, -1, :]  # [B,vocab_size]
            # take the last position in the sequence in order to predict the next token

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat(
                (idx, next_idx), dim=1
            )  # append the index we got to the sequence
        return idx


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load checkpoint
    ckpt = torch.load("ckpt.pt", map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    vocab_size = ckpt["vocab_size"]
    block_size = ckpt["block_size"]

    model = FullModel(
        vocab_size=vocab_size,
        nb_embd=nb_embd,
        block_size=block_size,
        nb_layers=nb_layers,
        nb_heads=nb_heads,
        embd_pdrop=embd_pdrop,
        residual_pdrop=residual_pdrop,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    def tokenize(text):

        tokens = [stoi[s] for s in text]
        return torch.tensor(tokens, dtype=torch.long)

    def tokens_to_string(tok):
        # tok will be a tensor of shape [B,N + max_new_tokens]
        # I will use as a seed only one batch so the output of the generation will be [1, N + max_new_tokens]

        string = [itos[i.item()] for i in tok[0]]
        return "".join(string)

    model.eval()
    with torch.no_grad():
        context = "O God, O God!"
        tokenized_context = tokenize(context).unsqueeze(0).to(device)
        y = model.generate(tokenized_context, 1600)
        completion = tokens_to_string(y)
        print(completion)


if __name__ == "__main__":
    main()

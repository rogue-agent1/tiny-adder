#!/usr/bin/env python3
"""
TinyAdder: 36-parameter hand-crafted transformer for 10-digit addition.
Credit: Alex Litzenberger (https://gist.github.com/alexlitz/0d5efbccf443fb0e8136b8f5bd85140a)

Parameter counting:
- Identity mappings (direct copy): 0 params
- Broadcast (1 value to N outputs): 1 param
- Distinct values: count each
"""

import torch
import torch.nn.functional as F
from math import log, exp

# === Constants ===
NUM_DIGITS = 10
TOKENS = [str(i) for i in range(NUM_DIGITS)] + ["=", "<bos>", "<eos>", "+"]
DIGIT_EMBED_SCALE = 10
V_SCALE = 1e4
DIGIT_SCALE = 1e10
FINAL_SCALE = 100
DIGIT_OFFSET = 0.5
GATE_BIAS_SHIFT = 15.0
ALIBI_CONSTANT = log(10)
EQ_DIM, SPECIAL_DIM, DIGIT_DIM, COUNT_DIM, SCALE_DIM = 0, 1, 2, 3, 4
EMBEDDING_DIM = 5
LAYER0_HEADS = 5
ADJUSTMENT_HEAD = 3
SCALE_HEAD = 4
CANDIDATES_START = 5
DIGIT_POS_DIM = 15
LAYER1_D_MODEL = 16
K_DIGIT_SCORE = -1000.0
K_SPECIAL_SCORE = -40.0
V_PROJ_SPECIAL = 0.1
V_PROJ_NEG_DOUBLE = -1.1
V_PROJ_SCALE = exp(K_SPECIAL_SCORE - log(10))

def softmax1(x, dim=-1):
    exp_x = x.exp()
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

def apply_alibi(seq_len, n_heads):
    pos = torch.arange(seq_len)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    slopes = torch.zeros(n_heads, dtype=torch.float64)
    slopes[ADJUSTMENT_HEAD] = ALIBI_CONSTANT
    return slopes.unsqueeze(1).unsqueeze(2) * rel_pos.unsqueeze(0)

def pad_to(x, d):
    if x.size(-1) >= d:
        return x[..., :d]
    return torch.cat([x, torch.zeros(*x.shape[:-1], d - x.size(-1), dtype=x.dtype)], dim=-1)

class TinyAdder:
    """36-parameter transformer for 10-digit addition.
    Params: 13 emb + 6 L0-attn + 12 L0-ffn + 2 L1-attn + 3 L1-ffn = 36
    """
    def __init__(self):
        d = torch.float64
        # === EMBEDDING (13 params) ===
        emb_idx = [[i, DIGIT_DIM] for i in range(1, 10)]
        emb_idx += [[10, EQ_DIM], [10, SPECIAL_DIM], [11, SPECIAL_DIM], [13, SPECIAL_DIM]]
        emb_val = [float(i * DIGIT_EMBED_SCALE) for i in range(1, 10)] + [1.0, 1.0, 1.0, 1.0]
        self.embedding = torch.sparse_coo_tensor(
            torch.tensor(emb_idx).T, torch.tensor(emb_val, dtype=d), (14, 5)
        ).to_dense()

        # === L0 ATTENTION (6 params) ===
        self.k0_weight = torch.tensor(K_SPECIAL_SCORE - K_DIGIT_SCORE, dtype=d)
        self.k0_bias = torch.tensor(K_DIGIT_SCORE, dtype=d)
        self.v0_w1 = torch.tensor(V_PROJ_SPECIAL / V_PROJ_SCALE, dtype=d)
        self.v0_w2 = torch.tensor(V_PROJ_NEG_DOUBLE / V_PROJ_SCALE, dtype=d)
        self.v0_w3 = torch.tensor(1.0, dtype=d)

        # === L0 FFN (12 params) ===
        pv = [(i + DIGIT_OFFSET) * DIGIT_SCALE * FINAL_SCALE for i in range(NUM_DIGITS)]
        self.up0_vals = torch.tensor(pv + [DIGIT_SCALE], dtype=d)

    @torch.inference_mode()
    def forward(self, x):
        batch_size, seq_len = x.shape
        d = torch.float64
        h = self.embedding[x]

        # === LAYER 0 ===
        h = pad_to(h, EMBEDDING_DIM)
        q = torch.ones(batch_size, seq_len, LAYER0_HEADS, dtype=d)
        k = torch.zeros(batch_size, seq_len, LAYER0_HEADS, dtype=d)
        k[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.k0_weight + self.k0_bias
        v = torch.zeros(batch_size, seq_len, LAYER0_HEADS, dtype=d)
        v[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.v0_w1 + h[..., EQ_DIM] * self.v0_w2
        v[..., SCALE_HEAD] = h[..., EQ_DIM] * self.v0_w3

        q = q.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)
        k = k.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)
        v = v.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) + apply_alibi(seq_len, LAYER0_HEADS).unsqueeze(0)
        scores = scores.masked_fill(torch.triu(torch.ones(seq_len, seq_len), 1).bool(), float('-inf'))
        attn = softmax1(scores, dim=-1).double()
        h = h + torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # FFN
        gate_in = torch.zeros(batch_size, seq_len, 11, dtype=d)
        gate_in[..., :NUM_DIGITS] = h[..., SCALE_DIM:SCALE_DIM+1]
        gate_in[..., NUM_DIGITS] = h[..., DIGIT_DIM]
        gate_out = F.relu(gate_in)
        up_out = h[..., COUNT_DIM:COUNT_DIM+1] * self.up0_vals
        ffn_hidden = gate_out * up_out
        h = pad_to(h, LAYER1_D_MODEL)
        h[..., 5:16] = h[..., 5:16] + ffn_hidden

        # === LAYER 1 ===
        q = torch.zeros(batch_size, seq_len, 1, dtype=d)
        k = torch.zeros(batch_size, seq_len, 1, dtype=d)
        v_weight = torch.zeros(LAYER1_D_MODEL, dtype=d)
        v_weight[DIGIT_POS_DIM] = FINAL_SCALE
        v = (h * v_weight).sum(dim=-1, keepdim=True) + GATE_BIAS_SHIFT

        q = q.view(batch_size, seq_len, 1, 1).transpose(1, 2)
        k = k.view(batch_size, seq_len, 1, 1).transpose(1, 2)
        v = v.view(batch_size, seq_len, 1, 1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores.masked_fill(torch.triu(torch.ones(seq_len, seq_len), 1).bool(), float('-inf'))
        attn = softmax1(scores, dim=-1).double()
        h = h + torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # FFN: V-shape
        candidates = h[..., CANDIDATES_START:CANDIDATES_START+NUM_DIGITS]
        gate_pos = F.relu(candidates * V_SCALE)
        gate_neg = F.relu(candidates * -V_SCALE)
        ffn_out = (gate_pos + gate_neg) * FINAL_SCALE
        h = pad_to(h, NUM_DIGITS)
        h = h + ffn_out

        return h.argmin(dim=-1)

def add(model, a: int, b: int) -> int:
    S = f"{a:010d}+{b:010d}="
    for i in range(11):
        toks = [TOKENS.index(t) for t in ["<bos>"] + list(S)]
        x = torch.tensor(toks).unsqueeze(0)
        pred = model.forward(x)
        next_digit = TOKENS[int(pred[0, -1].item())]
        S += next_digit
    return int("".join(list(S)[22:]))

if __name__ == "__main__":
    model = TinyAdder()
    print("TinyAdder: 36-parameter transformer for 10-digit addition")
    print("=" * 55)
    
    # Quick demo
    examples = [(1234567890, 9876543210), (9999999999, 1), (0, 0), (5555555555, 4444444445)]
    for a, b in examples:
        result = add(model, a, b)
        expected = a + b
        status = "✅" if result == expected else "❌"
        print(f"{status} {a} + {b} = {result} (expected {expected})")
    
    # Full test
    import random
    random.seed(42)
    correct = 0
    total = 1000
    for _ in range(total):
        a = random.randint(0, 9_999_999_999)
        b = random.randint(0, 9_999_999_999)
        if add(model, a, b) == a + b:
            correct += 1
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

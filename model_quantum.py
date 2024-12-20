import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


import pennylane as qml

class MultiHeadAttentionl(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_bias=False):
        super(MultiHeadAttentionl, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)



    def create_padding_mask(self, input_ids, padding_value=0):
        padding_mask = (input_ids == padding_value)
        return padding_mask

    def apply_padding_mask(self, scores, padding_mask, padding_value=float('-1e10')):
        masked_scores = scores.masked_fill(padding_mask, padding_value)
        return masked_scores

    def separate_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            padding_mask = self.create_padding_mask(mask)
            expanded_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = self.apply_padding_mask(scores, expanded_mask)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        attn = torch.matmul(scores, value)
        return attn, scores

    def downstream(self, query, key, value, batch_size, mask=None):
        Q = self.separate_heads(query)
        K = self.separate_heads(key)
        V = self.separate_heads(value)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return concat


    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        x = self.downstream(Q, K, V, batch_size, mask)
        output = self.combine_heads(x)
        return output

class QuanStrust(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(QuanStrust, self).__init__()
        self.n_qubits = 8          # Number of qubits used in the quantum component. Experiments show that increasing the number of qubits (4, 5, 6, 7, 8) improves performance.
        self.n_qlayers = 1
        self.linear_1 = nn.Linear(embed_dim, self.n_qubits)
        self.linear_2 = nn.Linear(embed_dim+self.n_qubits, embed_dim)
        self.q_device = "default.qubit"
        self.dev = qml.device(self.q_device, wires=self.n_qubits)
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (self.n_qlayers, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({self.n_qlayers}, {self.n_qubits})")
        self.q_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
    def forward(self, x):
        _, seq_len, _ = x.size()
        x_q = self.linear_1(x)

        x_q = [self.q_linear(x_q[:, t, :]) for t in range(seq_len)]
        x_q = torch.Tensor(pad_sequence(x_q))
        y = torch.cat([x_q, x], dim=2)
        y = self.linear_2(y)
        return y


class FeedForwardl(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForwardl, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, ffn_dim)
        self.linear_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlock, self).__init__()
        self.mask = None
        self.attn = MultiHeadAttentionl(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForwardl(embed_dim, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.quantum = QuanStrust(embed_dim)

    def forward(self, x):
        attn_output = self.attn(x, mask=self.mask)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)
        x = self.quantum(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim


        pe = torch.zeros(max_seq_len, embed_dim)

        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x * math.sqrt(self.embed_dim)

        seq_len = x.size(1)

        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x



class PolymerModel(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 vocab_size: int,
                 ffn_dim: int = 32):
        super(PolymerModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(embed_dim, 1)
        self.vocab_size = vocab_size


        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        print(f"++ There will be {num_blocks} transformer blocks")


        transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)
        ]

        self.transformers = nn.Sequential(*transformer_blocks)

    def forward(self, x, mask=None):
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.transformers:
            block.mask = mask
        x = self.transformers(x)
        return x


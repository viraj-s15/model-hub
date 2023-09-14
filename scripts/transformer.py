# Checking whether torch has been installed correctly along with ROCm

# Importing all necessary libraries

import copy
import math

import torch
import torch.matmul as matmul
import torch.nn as nn
import torch.optim as optim
import torch.softmax as softmax
import torch.utils.data as data

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# The above code returns "cuda:0" hence it is detecting and using our gpu

# Defining the multihead attention module of the nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init()
        assert (
            d_model % num_heads == 0
        ), "d_model must be completely divisible by num_head"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask=None):
        """
        scaled dot product is how we calculate the value of attention
        The formula for it is

        Attention = softmax((QK^T)/sqrt(dimensionality(k))) * V
        This is for the encoder, for the decoder, we will be using a mask
        we will simply add the mask in the formula. This gives us ->
        Attention = softmax((QK^T + mask)/sqrt(dimensionality(k))) * V
        """
        attention_scores = matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = softmax(attention_scores, dim=-1)
        output = matmul(attention_probs, V)
        return output

        def split_heads(self, x):
            """
            Reshaping the input tensor into multiple heads
            """
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(
                1, 2
            )

        def combine_heads(self, x):
            """
            Now that we have calculated the attention from all the heads,
            we must combine it again
            """
            batch_size, _, seq_length, d_k = x.size()
            return (
                x.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_length, self.d_model)
            )

        def forward(self, Q, K, V, mask=None):
            """
            generic function which defines the forward loop in this achitecture
            """
            Q = self.split_heads(self.W_q(Q))
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))

            attention = self.scaled_dot_product(Q, K, V, mask)
            return self.W_o(self.combine_heads(attention))


# The next step in creating a transformer is to feed the output
# of this layer into a simple feed forward neural network


class FeedForwardNn(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNn, self).__init()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.Relu()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


# We must now put this all together as a part of the encoder layer


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNn(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout

    def forward(self, x, mask):
        attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# We already have all the components to create a decoder layer as well


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNn(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, target_mask):
        attention = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention))
        attention = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


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


# We have all the classes needed to create a transformer


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        self(Transformer, self).__init__()
        self.encoder = nn.Embedding(src_vocab_size, d_model)
        self.decoder = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList(
            [Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        seq_len = target.size(1)

        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)
        ).bool()
        target_mask = target_mask & nopeak_mask
        return src_mask, target_mask

    def forward(self, src, target):
        src_mask, tgt_mask = self.generate_mask(src, target)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(target))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


if __name__ == "__main__":
    src_vocab_size = 1000
    target_vocab_size = 1000
    d_model = 512
    num_head = 8
    num_layers = 6
    d_ff = 1024
    max_seq_length = 100
    dropout = 0.01

    transformer = Transformer(
        src_vocab_size,
        target_vocab_size,
        d_model,
        num_head,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    src_data = torch.randint(1, src_vocab_size, src_vocab_size, (64, max_seq_length))
    target_data = torch.randint(
        1, target_vocab_size, src_vocab_size, (64, max_seq_length)
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.9), eps=1e-9
    )

    def train(num_epochs: int):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = transformer(src_data, target_data[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, target_vocab_size),
                target_data[:, :1].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            print(f"Epoch : {epoch}, Loss : {loss.item()}")

    train(250)

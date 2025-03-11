import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pickle

#############################
# Define the small music generator model
#############################

class SmallMusicGenerator(nn.Module):
    def __init__(self, n_vocab, sequence_length, embed_dim=128, num_heads=4, num_layers=3):
        """
        Initialize the SmallMusicGenerator model.
        :param n_vocab: Number of tokens in the vocabulary.
        :param sequence_length: Length of the input sequence.
        :param embed_dim: Dimension of the embeddings.
        :param num_heads: Number of attention heads in the Transformer encoder.
        :param num_layers: Number of Transformer encoder layers.
        """
        super(SmallMusicGenerator, self).__init__()
        # Embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # Register a buffer for the positional encoding; doubling the sequence length for some reason
        self.register_buffer("positional_encoding", self._generate_positional_encoding(sequence_length * 2, embed_dim))
        # Define a Transformer encoder with the specified parameters
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=256,  # Reduced size of the feed-forward network
                dropout=0.3,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        # Fully connected layer that maps the transformer output back to the vocabulary size
        self.fc = nn.Linear(embed_dim, n_vocab)

    def _generate_positional_encoding(self, seq_len, embed_dim):
        """
        Generate the positional encoding matrix.
        :param seq_len: Sequence length for which to generate the encoding.
        :param embed_dim: Embedding dimension.
        :return: Positional encoding tensor with shape (1, seq_len, embed_dim)
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(seq_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension
        return pos_enc.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass for the generator model.
        :param x: Input tensor of token indices with shape [B, sequence_length]
        :return: Output logits with shape [B, sequence_length, n_vocab]
        """
        embedded = self.embedding(x)  # Convert token indices to embeddings: [B, sequence_length, embed_dim]
        seq_len = x.size(1)
        # Extract positional encoding for the current sequence length
        pos_enc = self.positional_encoding[:, :seq_len, :]
        x = embedded + pos_enc  # Add positional information to the embeddings
        x = self.transformer(x)  # Pass through the Transformer encoder
        x = self.fc(x)           # Map to vocabulary logits
        return x

#############################
# Define the small music discriminator model
#############################

class SmallMusicDiscriminator(nn.Module):
    def __init__(self, n_vocab, sequence_length, embed_dim=128):
        """
        Initialize the SmallMusicDiscriminator model.
        :param n_vocab: Number of tokens in the vocabulary.
        :param sequence_length: Length of the input sequence.
        :param embed_dim: Dimension of the embeddings.
        """
        super(SmallMusicDiscriminator, self).__init__()
        # Embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # Fully connected network for discrimination
        # Note: No sigmoid is applied because BCEWithLogitsLoss is expected to be used
        self.fc = nn.Sequential(
            nn.Linear(sequence_length * embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass for the discriminator model.
        :param x: Input tensor of token indices with shape [B, sequence_length]
        :return: Output logits with shape [B, 1]
        """
        x = self.embedding(x)  # Convert token indices to embeddings: [B, sequence_length, embed_dim]
        x = x.view(x.size(0), -1)  # Flatten the tensor to shape [B, sequence_length * embed_dim]
        x = self.fc(x)  # Pass through the fully connected network
        return x

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pickle

###########################
# הגדרת המודלים המוקטנים
###########################

class SmallMusicGenerator(nn.Module):
    def __init__(self, n_vocab, sequence_length, embed_dim=128, num_heads=4, num_layers=3):
        super(SmallMusicGenerator, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # נבנה positional encoding עבור אורך כפול מהקלט
        self.register_buffer("positional_encoding", self._generate_positional_encoding(sequence_length * 2, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=256,  # הקטנה של ה-FFN
                dropout=0.3,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, n_vocab)

    def _generate_positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(seq_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)  # מוסיף ממד batch

    def forward(self, x):
        embedded = self.embedding(x)  # [B, sequence_length, embed_dim]
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :]
        x = embedded + pos_enc
        x = self.transformer(x)
        x = self.fc(x)  # [B, sequence_length, n_vocab]
        return x

class SmallMusicDiscriminator(nn.Module):
    def __init__(self, n_vocab, sequence_length, embed_dim=128):
        super(SmallMusicDiscriminator, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(sequence_length * embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # אין Sigmoid כי נשתמש ב-BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.embedding(x)  # [B, sequence_length, embed_dim]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # [B, 1]
        return x
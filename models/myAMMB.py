import os

import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
import math

from torch.autograd import Variable

DEBUG = True


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super(TransformerEncoder, self).__init__()

        self.layerNorm1 = nn.LayerNorm(embed_dim)
        self.layerNorm2 = nn.LayerNorm(embed_dim)

        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.FFN = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x1 = self.self_attention(x, x, x, attn_mask=None, key_padding_mask=None)[0]
        x1 = self.dropout1(x1)
        x1 = self.layerNorm1(x1 + x)

        x2 = self.FFN(x1)
        x2 = self.dropout2(x2)
        x2 = self.layerNorm2(x2 + x1)

        return x2


class Positional_Encoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(Positional_Encoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)  # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0)  # batch, channels, seq_len

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()

        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]

        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]

        x = self.dropout(x)

        x = x.permute(0, 2, 1).contiguous()

        return x


# AmbiMiMo
#  Sequenza di transformer interchannel, intrachunk e interchunk
class AmbiMiMo(nn.Module):
    def __init__(self, num_heads, embed_dim, num_encoders):

        super(AmbiMiMo, self).__init__()

        self.num_encoders = num_encoders

        self.interChaPE = Positional_Encoding(d_model=embed_dim, max_len=32000)
        self.intraChuPE = Positional_Encoding(d_model=embed_dim, max_len=32000)
        self.interChuPE = Positional_Encoding(d_model=embed_dim, max_len=32000)

        self.interchannel_block = nn.ModuleList([])
        self.intrachunk_block = nn.ModuleList([])
        self.interchunk_block = nn.ModuleList([])

        for i in range(num_encoders):
            self.interchannel_block.append(TransformerEncoder(num_heads=num_heads, embed_dim=embed_dim, dropout=0.1))
            self.intrachunk_block.append(TransformerEncoder(num_heads=num_heads, embed_dim=embed_dim, dropout=0.1))
            self.interchunk_block.append(TransformerEncoder(num_heads=num_heads, embed_dim=embed_dim, dropout=0.1))

    def forward(self, x):

        print("Dentor AmbiMiMo") if DEBUG else None
        print(x.shape) if DEBUG else None
        B, Ch, NBin, C, NC = x.shape
        x_interCh = x.permute(0, 3, 4, 1, 2).contiguous().view(B * C * NC, Ch, NBin)
        x_interCh = x_interCh + self.interChaPE(x_interCh)
        print(x_interCh.shape) if DEBUG else None
        for i in range(self.num_encoders):
            x_interCh = self.interchannel_block[i](x_interCh)
        print(x_interCh.shape) if DEBUG else None

        x_interCh = x_interCh.view(B, C, NC, Ch, NBin)

        x_intraChu = x_interCh.permute(0, 3, 2, 1, 4).contiguous().view(B * Ch * NC, C, NBin)
        x_intraChu = x_intraChu + self.intraChuPE(x_intraChu)
        print(x_intraChu.shape) if DEBUG else None
        for i in range(self.num_encoders):
            x_intraCha = self.intrachunk_block[i](x_intraChu)
        print(x_intraChu.shape) if DEBUG else None

        x_intraChu = x_intraChu.view(B, Ch, NC, C, NBin)

        x_interChu = x_intraChu.permute(0, 1, 3, 2, 4).contiguous().view(B * Ch * C, NC, NBin)
        x_interChu = x_interChu + self.interChuPE(x_interChu)
        print(x_interChu.shape) if DEBUG else None
        for i in range(self.num_encoders):
            x_interChu = self.intrachunk_block[i](x_interChu)
        print(x_interChu.shape) if DEBUG else None

        x_interChu = x_interChu.view(B, Ch, C, NC, NBin).permute(0, 1, 4, 2, 3).contiguous()

        return x_interChu


# classe AMMB
# forward che implementa STFT, chunking, AmbiMiMo, Somma, iSTFT
class AMMB(nn.Module):
    def __init__(self, num_heads, embed_dim, num_encoders, global_B, channel_dim, fft_size, hop_size, win_size, dropout,
                 device):

        super(AMMB, self).__init__()
        self.channel_dim = channel_dim
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.device = device
        self.global_B = global_B

        self.width_chunk = 16
        self.dim_chunk = 512
        self.valid_freq = int(fft_size // 2)

        self.chunking = nn.Conv2d(1, self.dim_chunk, (self.fft_size, self.width_chunk),
                                  stride=(self.fft_size, self.width_chunk // 2))

        self.extend_chunking = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 16)
        )

        self.AmbiMiMo = nn.ModuleList([])
        for i in range(global_B):
            self.AmbiMiMo.append(AmbiMiMo(num_heads, embed_dim, num_encoders))

        self.out = nn.Linear(1248, 600)  # Prima 1184

    def extract_features(self, inputs, device):
        # shape: [B, C, S]
        batch_size, channel, samples = inputs.size()

        features = []
        for idx in range(batch_size):
            # shape: [C, F, T, 2]
            features_batch = torch.stft(
                inputs[idx, ...],
                self.fft_size,
                self.hop_size,
                self.win_size,
                torch.hann_window(self.win_size).to(device),
                pad_mode='constant',
                onesided=True,
                return_complex=False)
            features.append(features_batch)

        # shape: [B, C, F, T, 2]
        features = torch.stack(features, 0)
        features = features[:, :, :self.valid_freq, :, :]
        real_features = features[..., 0]
        imag_features = features[..., 1]

        return real_features, imag_features

    def pad_segment(self, input, segment_size):

        # Input shape: (B, N, T)

        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):

        # Chunking
        # Input shape: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2,
                                                                                                          3).contiguous()

        return segments, rest

    def forward(self, x, device):
        # STFT
        print(f"Input dim {x.shape}") if DEBUG else None
        real_features, imag_features = self.extract_features(x, self.device)  # [B, Ch, NBin, TimeFrames]
        x_stft = torch.cat((real_features, imag_features), 2)

        print(x_stft.shape) if DEBUG else None
        # Chunking
        # x_c = x_stft.unsqueeze(2).view(-1, 1, x_stft.shape[2], x_stft.shape[3])  # [B*Ch, 1, NBin, TF]
        # x_c = self.chunking(x_c) # [B*Ch, NBin, C, NC]
        x_c = x_stft.unsqueeze(2).view(-1, x_stft.shape[2], x_stft.shape[3])
        x_c, gap = self.split_feature(x_c, 16)

        # x_c = x_c.view(x.shape[0], 4, self.dim_chunk, x_c.shape[-2], 1)

        # x_c = x_c.transpose(-1, -2).view(x.shape[0], x.shape[1], x_c.shape[1], x_c.shape[3], 1)  # [B, Ch, NBin, NC, 1]
        x_c = x_c.view(x.shape[0], x.shape[1], x_c.shape[1], x_c.shape[3], x_c.shape[2])
        # x_c = self.extend_chunking(x_c).transpose(-1, -2)     # [B, Ch, NBin, C, NC]
        print(x_c.shape) if DEBUG else None

        # AmbiMiMo
        for i in range(self.global_B):
            x_mimo = self.AmbiMiMo[i](x_c)

        # print(x_c.shape)

        print(f"x_mimo {x_mimo.shape}") if DEBUG else None
        x_mimo = x_mimo.view(x_mimo.shape[0], x_mimo.shape[1], x_mimo.shape[2], x_mimo.shape[3] * x_mimo.shape[4])
        x_mimo = self.out(x_mimo)

        print(f"x_mimo {x_mimo.shape}") if DEBUG else None

        real_mask = x_mimo[:, :, :self.valid_freq, :]
        imag_mask = x_mimo[:, :, self.valid_freq:, :]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)

        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)

        print(f"est_speech_stft {est_speech_stft.shape}") if DEBUG else None

        # shape: [B, S]
        est_speech = torch.istft(
            est_speech_stft,
            self.fft_size,
            self.hop_size,
            self.win_size,
            torch.hann_window(self.win_size).to(device))
        # shape: [B, 1, S]
        print(f"est_speech {est_speech.shape}") if DEBUG else None
        return est_speech.view(1, -1, est_speech.shape[-1])


if __name__ == '__main__':
    device = 'cpu'
    x = torch.empty((1, 4, 512, 600))
    ammb = AMMB(4, 512, 2, 4, 512, 0, 0, 0.1, device)
    x = ammb(x, device)
    print(x.shape)
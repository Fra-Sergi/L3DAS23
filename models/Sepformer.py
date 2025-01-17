import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, L, N):
        """
            Apprendimento di una rappresentazione simile a STFT。
            Lo stride factor della convoluzione ha un impatto significativo su prestazioni, velocità e memoria del modello.
        """

        super(Encoder, self).__init__()

        self.L = L  # Dimensione Kernel di convoluzione

        self.N = N  # dimensione del canale di uscita

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=N,
                                kernel_size=L,
                                stride=L // 2,
                                padding=0,
                                bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.Conv1d(x)

        x = self.ReLU(x)

        return x


class Decoder(nn.Module):

    def __init__(self, L, N):
        super(Decoder, self).__init__()

        self.L = L

        self.N = N

        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=N,
                                                  out_channels=1,
                                                  kernel_size=L,
                                                  stride=L // 2,
                                                  padding=0,
                                                  bias=False)

    def forward(self, x):
        x = self.ConvTranspose1d(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, num_hiddens=512):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_hiddens, kernel_size=(512, 16), stride=8)

    def forward(self, X):
        # output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).transpose(1, 2).view(1, 4, 512, 74)


class TransformerEncoderLayer(Module):
    """
        TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application.

        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of intermediate layer, relu or gelu (default=relu).

        Examples:
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> src = torch.rand(10, 32, 512)
            >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0):
        super(TransformerEncoderLayer, self).__init__()

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.Dropout1 = nn.Dropout(p=dropout)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=d_model)

        self.FeedForward = nn.Sequential(nn.Linear(d_model, d_model * 2 * 2),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(d_model * 2 * 2, d_model))

        self.Dropout2 = nn.Dropout(p=dropout)

    def forward(self, z):
        z1 = self.LayerNorm1(z)

        z2 = self.self_attn(z1, z1, z1, attn_mask=None, key_padding_mask=None)[0]

        z3 = self.Dropout1(z2) + z

        z4 = self.LayerNorm2(z3)

        z5 = self.Dropout2(self.FeedForward(z4)) + z3

        return z5


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
        #print("pos_ecn: ", x.shape)
        x = x + self.pe[:, :, :x.size(2)]

        x = self.dropout(x)

        x = x.permute(0, 2, 1).contiguous()

        return x


class DPTBlock(nn.Module):

    def __init__(self, input_size, nHead, Local_B):

        super(DPTBlock, self).__init__()

        self.Local_B = Local_B

        self.channel_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.channel_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.channel_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                    nhead=nHead,
                                                                    dropout=0.1))

        self.intra_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.intra_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.intra_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0.1))

        self.inter_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=32000)
        self.inter_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.inter_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0.1))

    def forward(self, z):

        A, B, N, K, P = z.shape
        #if __debug__:
            #print("in_trasf: ", z.shape)
        # interchannel DPT
        prova_z = z.permute(0, 3, 4, 1, 2).contiguous().view(A * K * P, B, N)
        #if __debug__:
         #   print("in_trasf2: ", prova_z.shape)
        prova_z1 = self.channel_PositionalEncoding(prova_z)

        for i in range(self.Local_B):
            prova_z1 = self.channel_transformer[i](prova_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        prova_f = prova_z1 + prova_z
        prova_output = prova_f.view(A, K, P, B, N).permute(0, 3, 4, 1, 2).contiguous()

        # intra DPT
        row_z = prova_output.permute(0, 1, 4, 3, 2).contiguous().view(A * B * P, K, N)
        row_z1 = self.intra_PositionalEncoding(row_z)

        for i in range(self.Local_B):
            row_z1 = self.intra_transformer[i](row_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        row_f = row_z1 + row_z
        row_output = row_f.view(A, B, P, K, N).permute(0, 1, 4, 3, 2).contiguous()

        # inter DPT
        col_z = row_output.permute(0, 1, 3, 4, 2).contiguous().view(A * B * K, P, N)
        col_z1 = self.inter_PositionalEncoding(col_z)

        for i in range(self.Local_B):
            col_z1 = self.inter_transformer[i](col_z1.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        col_f = col_z1 + col_z
        col_output = col_f.view(A, B, K, P, N).permute(0, 1, 4, 2, 3).contiguous()

        return col_output


class Separator(nn.Module):

    def __init__(self, N, H, Global_B, Local_B):

        super(Separator, self).__init__()

        self.N = N
        # self.C = C
        # self.K = K
        self.Linear = nn.Linear(74, 74 * 16)
        self.Global_B = Global_B  # 全局循环次数
        self.Local_B = Local_B  # 局部循环次数

        self.patchEmbedding = PatchEmbedding()
        self.LayerNorm = nn.LayerNorm(self.N)
        self.Linear1 = nn.Linear(in_features=self.N, out_features=self.N, bias=None)

        self.SepFormer = nn.ModuleList([])
        for i in range(self.Global_B):
            self.SepFormer.append(DPTBlock(N, H, self.Local_B))

        # self.PReLU = nn.PReLU()
        # self.Conv2d = nn.Conv2d(N, N * C, kernel_size=1)

        # self.output = nn.Sequential(nn.Conv1d(N, N, 1), nn.Tanh())
        # self.output_gate = nn.Sequential(nn.Conv1d(N, N, 1), nn.Sigmoid())

    def forward(self, x):
        # Norm + Linear
        # x = self.LayerNorm(x.permute(0, 2, 1).contiguous())  # [B, C, L] => [B, L, C]
        # x = self.Linear1(x).permute(0, 2, 1).contiguous()  # [B, L, C] => [B, C, L]

        # Chunking
        # out, gap = self.split_feature(x, self.K)  # [B, C, L] => [B, C, K, S]
        out = self.patchEmbedding(x)  # [B, 4, 512, 600] => [B, 4, 512, 74]
        out = self.Linear(out)
        out = out.view(1, 4, 512, 16, 74)
        # SepFormer
        for i in range(self.Global_B):
            out = self.SepFormer[i](out)  # [B, C, K, S]

        # out = self.Conv2d(self.PReLU(out))  # [B, N, K, S] -> [B, N*C, K, S], torch.Size([1, 128, 250, 130])

        # B, _, K, S = out.shape
        # out = out.view(B, -1, self.C, K, S).permute(0, 2, 1, 3, 4).contiguous()  # [B, N*C, K, S] -> [B, N, C, K, S]
        # out = out.view(B * self.C, -1, K, S)
        # out = self.merge_feature(out, gap)  # [B*C, N, K, S]  -> [B*C, N, L]

        # out = F.relu(self.output(out) * self.output_gate(out))
        # out = F.relu(out)

        return out

    def pad_segment(self, input, segment_size):

        # 输入特征: (B, N, T)

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

        # 将特征分割成段大小的块
        # 输入特征: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2,
                                                                                                          3).contiguous()

        return segments, rest

    def merge_feature(self, input, rest):

        # 将分段的特征合并成完整的话语
        # 输入特征: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2

        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T


class Sepformer(nn.Module):
    """
        Args:
            C: Number of speakers
            N: Number of filters in autoencoder
            L: Length of the filters in autoencoder
            H: Multi-head
            K: segment size
            R: Number of repeats
    """

    def __init__(self, N=64, C=2, L=4, H=4, K=250, Global_B=3, Local_B=5):

        super(Sepformer, self).__init__()

        self.N = N  # Number of filters in autoencoder
        self.C = C  # Number of speakers
        self.L = L  # Length of the filters in autoencoder
        self.H = H  # Multi-head
        self.K = K  # Number of repeats
        self.Global_B = Global_B  # 全局循环次数
        self.Local_B = Local_B  # 局部循环次数

        self.encoder = Encoder(self.L, self.N)

        self.separator = Separator(self.N, self.C, self.H, self.K, self.Global_B, self.Local_B)

        self.decoder = Decoder(self.L, self.N)

    def forward(self, x):

        # Encoding
        x, rest = self.pad_signal(x)  # 补零，torch.Size([1, 1, 32006])

        enc_out = self.encoder(x)  # [B, 1, T] -> [B, N, I]，torch.Size([1, 64, 16002])

        # Mask estimation
        masks = self.separator(enc_out)  # [B, N, I] -> [B*C, N, I]，torch.Size([2, 64, 16002])

        _, N, I = masks.shape

        masks = masks.view(self.C, -1, N, I)  # [C, B, N, I]，torch.Size([2, 1, 64, 16002])

        # Masking
        out = [masks[i] * enc_out for i in range(self.C)]  # C * ([B, N, I]) * [B, N, I]

        # Decoding
        audio = [self.decoder(out[i]) for i in range(self.C)]  # C * [B, 1, T]

        audio[0] = audio[0][:, :, self.L // 2:-(rest + self.L // 2)].contiguous()  # B, 1, T
        audio[1] = audio[1][:, :, self.L // 2:-(rest + self.L // 2)].contiguous()  # B, 1, T
        audio = torch.cat(audio, dim=1)  # [B, C, T]

        return audio

    def pad_signal(self, input):

        # 输入波形: (B, T) or (B, 1, T)
        # 调整和填充

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)  # 每一个批次的大小
        nsample = input.size(2)  # 单个数据的长度
        rest = self.L - (self.L // 2 + nsample % self.L) % self.L

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.L // 2)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    @classmethod
    def load_model(cls, path):

        package = torch.load(path, map_location=lambda storage, loc: storage)

        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(N=package['N'], C=package['C'], L=package['L'],
                    H=package['H'], K=package['K'], Global_B=package['Global_B'],
                    Local_B=package['Local_B'])

        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):

        package = {
            # hyper-parameter
            'N': model.N, 'C': model.C, 'L': model.L,
            'H': model.H, 'K': model.K, 'Global_B': model.Global_B,
            'Local_B': model.Local_B,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package


class MIMO(nn.Module):
    def __init__(self,
                 fft_size=512,
                 hop_size=128,
                 N=512,
                 C=4,
                 L=4,
                 H=4,
                 K=250,
                 Global_B=3,
                 Local_B=5
                 ):
        super(MIMO, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = fft_size
        self.valid_freq = int(self.fft_size / 2)

        self.N = N  # Number of filters in autoencoder
        self.C = C  # Number of speakers
        self.L = L  # Length of the filters in autoencoder
        self.H = H  # Multi-head
        self.K = K  # Number of repeats
        self.Global_B = Global_B  # 全局循环次数
        self.Local_B = Local_B  # 局部循环次数
        self.separator = Separator(self.N, self.H, self.Global_B, self.Local_B)
        self.linear = nn.Linear(74*16, 600)
        # layer_number = len(unet_channel)
        # kernel_number = len(kernel_size)
        # stride_number = len(stride)
        # assert layer_number==kernel_number==stride_number

        # self.kernel = kernel_size
        # self.stride = stride

        # encoder setting
        # self.encoder = nn.ModuleList()
        # self.encoder_channel = [input_channel] + unet_channel

        # decoder setting
        # self.decoder = nn.ModuleList()
        # self.decoder_outchannel = unet_channel
        # self.decoder_inchannel = list(map(lambda x:x[0] + x[1] ,zip(unet_channel[1:] + [0], unet_channel)))

        # self.conv2d = nn.Conv2d(self.decoder_outchannel[0], input_channel, 1, 1)
        # self.linear = nn.Linear(self.valid_freq * 2, self.valid_freq * 2)

        # for idx in range(layer_number):
        #   self.encoder.append(
        #      nn.Sequential(
        #         nn.Conv2d(
        #            self.encoder_channel[idx],
        #            self.encoder_channel[idx+1],
        #            self.kernel[idx],
        #            self.stride[idx],
        #        ),
        #        nn.BatchNorm2d(self.encoder_channel[idx+1]),
        #        nn.LeakyReLU(0.3)
        #    )
        # )

        # for idx in range(layer_number):
        #    self.decoder.append(
        #        nn.Sequential(
        #            nn.ConvTranspose2d(
        #                self.decoder_inchannel[-1-idx],
        #                self.decoder_outchannel[-1-idx],
        #                self.kernel[-1-idx],
        #                self.stride[-1-idx]
        #            ),
        #            nn.BatchNorm2d(self.decoder_outchannel[-1-idx]),
        #            nn.LeakyReLU(0.3)
        #        )
        #    )

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

    # def encode_padding_size(self, kernel_size):
    #    k_f, k_t = kernel_size
    #    p_t_s = int(k_t / 2)
    #    p_f_s = int(k_f / 2)

    #    p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

    #    if k_t % 2 == 0:
    #        p_t_0 = p_t_0 - 1

    #    if k_f % 2 == 0:
    #        p_f_0 = p_f_0 - 1

    #    return (p_t_0, p_t_1, p_f_0, p_f_1)

    # def decode_padding_size(self, in_size, target_size):
    #    i_f, i_t = in_size
    #    t_f, t_t = target_size
    #    p_t_s = int(abs(t_t - i_t) / 2)
    #    p_f_s = int(abs(t_f - i_f) / 2)

    #    p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

    #    if abs(t_t - i_t) % 2 == 1:
    #        p_t_1 = p_t_1 + 1

    #    if abs(t_f - i_f) % 2 == 1:
    #        p_f_1 = p_f_1 + 1

    #    return (p_t_0, p_t_1, p_f_0, p_f_1)

    # def encode_padding_same(self, features, kernel_size):
    #    p_t_0, p_t_1, p_f_0, p_f_1 = self.encode_padding_size(kernel_size)

    #    features = F.pad(features, (p_t_0, p_t_1, p_f_0, p_f_1))

    #    return features

    # def decode_padding_same(self, features, encoder_features, stride):
    # shape: [B, C, F, T]
    #    _, _, f, t = features.size()
    #    _, _, ef, et = encoder_features.size()

    # shape: [F, T]
    #    sf, st = stride
    #    tf, tt = (int(ef * sf), int(et * st))

    #    p_t_0, p_t_1, p_f_0, p_f_1 = self.decode_padding_size((f, t), (tf, tt))

    # shape: [B, C, F, T]
    #    if (p_t_0 != 0) or (p_t_1 != 0):
    #        features = features[:, :, :, p_t_0:-p_t_1]
    #    if (p_f_0 != 0) or (p_f_1 != 0):
    #        features = features[:, :, p_f_0:-p_f_1, :]

    #    return features

    def forward(self, inputs, device):
        # shape: [B, C, F, T]
        real_features, imag_features = self.extract_features(inputs, device)
        # shape: [B, C, F*2, T]
        features = torch.cat((real_features, imag_features), 2)

        out = features
        out1 = torch.unsqueeze(out, 2)
        out2 = out1.view(1 * 4, 1, 512, 600)
        #if __debug__:
         #   print("shape pre mask: ", out.shape)
        masks = self.separator(out2)
        masks = masks.view(1, 4, 512, 16 * 74)
        masks = self.linear(masks)
        #if __debug__:
            #print("mask_out: ", masks.shape)
        # encoder_out = []
        # for idx, layer in enumerate(self.encoder):
        #    out = self.encode_padding_same(out, self.kernel[idx])
        #    out = layer(out)
        #    encoder_out.append(out)

        # out = encoder_out[-1]
        # for idx, layer in enumerate(self.decoder):
        #    if idx != 0:
        #        out = torch.cat((out, encoder_out[-1-idx]), 1)
        #    out = layer(out)
        #    out = self.decode_padding_same(out, encoder_out[-1-idx], self.stride[-1-idx])

        # out = self.conv2d(out)
        # shape: [B, C, T, F*2]
        # out = out.permute(0,1,3,2)
        # out = self.linear(out)
        # shape: [B, C, F*2, T]
        # out = out.permute(0,1,3,2)

        real_mask = masks[:,:,:self.valid_freq,:]
        imag_mask = masks[:,:,self.valid_freq:,:]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)

        #est_speech_stft = torch.mul(out, masks)
        #if __debug__:
         #   print("est_speech: ", est_speech_stft.shape)
        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)

        # shape: [B, S]
        est_speech = torch.istft(
            est_speech_stft,
            self.fft_size,
            self.hop_size,
            self.win_size,
            torch.hann_window(self.win_size).to(device))
        # shape: [B, 1, S]
        return torch.unsqueeze(est_speech, 1)


if __name__ == '__main__':
    '''
    The frame number input to the model must be a multiple of 8, here it's 600.
    Because the torch.stft pads 4 extra frames based on our configurations, the
    frame number of the signal is 596 actually, ie. the duration of the signal
    is 4.792 seconds(76672 sample), while the fft_size is 512, hop_size is 128,
    and the sample_rate is 16000.
    The frequency bin input to the model must be a multiple of 16, here it's 256.
    '''
    frames_num = 600
    fft_size = 512
    hop_size = 128
    batch_size = 1
    audio_channel = 4
    length = int((frames_num - 1) * hop_size + fft_size - 4 * hop_size)  # 4.792 seconds
    inputs = torch.rand(batch_size, audio_channel, length)

    model = MIMO()
    out = model(inputs, 'cpu')
    print('input size:', inputs.size())
    print('out size:', out.size())

    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ' + str(model_params))

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
A pytorch implementation of the neural beamforming network described in 'A NEURAL BEAMFORMING NETWORK FOR B-FORMAT 3D SPEECH ENHANCEMENT AND RECOGNITION'
'''

class MIMO_UNet_Beamforming(nn.Module):
    def __init__(self,
                fft_size=512,
                hop_size=128,
                input_channel=4, # the channel number of input audio
                unet_channel=[32,32,32,64,64,96,96,96,128,256],
                kernel_size=[(7,1),(1,7),(8,6),(7,6),(6,5),(5,5),(6,3),(5,3),(6,3),(5,3)],
                stride=[(1,1),(1,1),(2,2),(1,1),(2,2),(1,1),(2,2),(1,1),(2,1),(1,1)]
                ):
        super(MIMO_UNet_Beamforming, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = fft_size
        self.valid_freq = int(self.fft_size / 2)

        layer_number = len(unet_channel)
        kernel_number = len(kernel_size)
        stride_number = len(stride)
        assert layer_number==kernel_number==stride_number

        self.kernel = kernel_size
        self.stride = stride

        # encoder setting
        self.encoder = nn.ModuleList()
        self.encoder_channel = [input_channel] + unet_channel

        # decoder setting
        self.decoder = nn.ModuleList()
        self.decoder_outchannel = unet_channel
        self.decoder_inchannel = list(map(lambda x:x[0] + x[1] ,zip(unet_channel[1:] + [0], unet_channel)))

        self.conv2d = nn.Conv2d(self.decoder_outchannel[0], input_channel, 1, 1)
        self.linear = nn.Linear(self.valid_freq * 2, self.valid_freq * 2)

        for idx in range(layer_number):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.encoder_channel[idx],
                        self.encoder_channel[idx+1],
                        self.kernel[idx],
                        self.stride[idx],
                    ),
                    nn.BatchNorm2d(self.encoder_channel[idx+1]),
                    nn.LeakyReLU(0.3)
                )
            )

        for idx in range(layer_number):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_inchannel[-1-idx],
                        self.decoder_outchannel[-1-idx],
                        self.kernel[-1-idx],
                        self.stride[-1-idx]
                    ),
                    nn.BatchNorm2d(self.decoder_outchannel[-1-idx]),
                    nn.LeakyReLU(0.3)
                )
            )

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
        features = features[:,:,:self.valid_freq,:,:]
        real_features = features[..., 0]
        imag_features = features[..., 1]

        return real_features, imag_features

    def encode_padding_size(self, kernel_size):
        k_f, k_t = kernel_size
        p_t_s = int(k_t / 2)
        p_f_s = int(k_f / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if k_t % 2 == 0:
            p_t_0 = p_t_0 - 1

        if k_f % 2 == 0:
            p_f_0 = p_f_0 - 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def decode_padding_size(self, in_size, target_size):
        i_f, i_t = in_size
        t_f, t_t = target_size
        p_t_s = int(abs(t_t - i_t) / 2)
        p_f_s = int(abs(t_f - i_f) / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if abs(t_t - i_t) % 2 == 1:
            p_t_1 = p_t_1 + 1

        if abs(t_f - i_f) % 2 == 1:
            p_f_1 = p_f_1 + 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def encode_padding_same(self, features, kernel_size):
        p_t_0, p_t_1, p_f_0, p_f_1 = self.encode_padding_size(kernel_size)

        features = F.pad(features, (p_t_0, p_t_1, p_f_0, p_f_1))

        return features

    def decode_padding_same(self, features, encoder_features, stride):
        # shape: [B, C, F, T]
        _, _, f, t = features.size()
        _, _, ef, et = encoder_features.size()

        # shape: [F, T]
        sf, st = stride
        tf, tt = (int(ef * sf), int(et * st))

        p_t_0, p_t_1, p_f_0, p_f_1 = self.decode_padding_size((f, t), (tf, tt))

        # shape: [B, C, F, T]
        if (p_t_0 != 0) or (p_t_1 != 0):
            features = features[:, :, :, p_t_0:-p_t_1]
        if (p_f_0 != 0) or (p_f_1 != 0):
            features = features[:, :, p_f_0:-p_f_1, :]

        return features

    def forward(self, inputs, device):
        # shape: [B, C, F, T]
        real_features, imag_features = self.extract_features(inputs, device)
        # shape: [B, C, F*2, T]
        features = torch.cat((real_features, imag_features), 2)

        out = features
        #print("Number of dimensions of out=features: ", out.ndim)
        #print("Shape of out=features:", out.shape)
        encoder_out = []
        for idx, layer in enumerate(self.encoder):
            out = self.encode_padding_same(out, self.kernel[idx])
            out = layer(out)
            encoder_out.append(out)

        out = encoder_out[-1]
        for idx, layer in enumerate(self.decoder):
            if idx != 0:
                out = torch.cat((out, encoder_out[-1-idx]), 1)
            out = layer(out)
            out = self.decode_padding_same(out, encoder_out[-1-idx], self.stride[-1-idx])

        out = self.conv2d(out)
        # shape: [B, C, T, F*2]
        out = out.permute(0,1,3,2)
        out = self.linear(out)
        # shape: [B, C, F*2, T]
        out = out.permute(0,1,3,2)

        real_mask = out[:,:,:self.valid_freq,:]
        imag_mask = out[:,:,self.valid_freq:,:]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)

        # shape: [B, C, F, T]
        print("Number of dimensions of est_speech_stft_1: ", est_speech_stft.ndim)
        print("Shape of dimensions of est_speech_stft_1:", est_speech_stft.shape)
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)
        print("Number of dimensions of est_speech_stft_2: ", est_speech_stft.ndim)
        print("Shape of dimensions of est_speech_stft_2:", est_speech_stft.shape)

        # shape: [B, S]
        est_speech = torch.istft(
                        est_speech_stft,
                        self.fft_size,
                        self.hop_size,
                        self.win_size,
                        torch.hann_window(self.win_size).to(device))
        # shape: [B, 1, S]
        print("Number of dimensions of est_speech: ", est_speech.ndim)
        print("Shape of dimensions of est_speech:", est_speech.shape)
        return torch.unsqueeze(est_speech, 1)


class FiLMBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLMBlock, self).__init__()
        self.beta = nn.Linear(in_features, out_features)
        self.gamma = nn.Linear(in_features, out_features)

    def forward(self, x, visual_features):
        
        # beta = self.beta.view(x.size(0), x.size(1), 1, 1)
        # gamma = self.gamma.view(x.size(0), x.size(1), 1, 1)
        gamma = self.gamma(visual_features).view(x.size(0), x.size(1), 1, 1)
        beta = self.beta(visual_features).view(x.size(0), x.size(1), 1, 1)
        x = gamma * x + beta
        # [12, 32, 512, 600]
        # [12, 300, 32] 12, 32
        return x

class audiovisual_MIMO_UNet_Beamforming(nn.Module):
    def __init__(self,
                fft_size=512,
                hop_size=128,
                input_channel=4, # the channel number of input audio
                unet_channel=[32,32,32,64,64,96,96,96,128,256],
                kernel_size=[(7,1),(1,7),(8,6),(7,6),(6,5),(5,5),(6,3),(5,3),(6,3),(5,3)],
                stride=[(1,1),(1,1),(2,2),(1,1),(2,2),(1,1),(2,2),(1,1),(2,1),(1,1)],
                pool_size_visual=[2,2,2,2,2,2], cnn_filters_visual=[32,32,64,128,128,256],
                dropout_visual = .3
                ):
        super(audiovisual_MIMO_UNet_Beamforming, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = fft_size
        self.valid_freq = int(self.fft_size / 2)

        layer_number = len(unet_channel)
        kernel_number = len(kernel_size)
        stride_number = len(stride)
        assert layer_number==kernel_number==stride_number

        self.kernel = kernel_size
        self.stride = stride

        # image feature extraction
        conv_layers_visual = []
        in_chans = 3
        for i, (p,c) in enumerate(zip(pool_size_visual, cnn_filters_visual)):
            curr_chans = c

            conv_layers_visual.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans,
                                kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.MaxPool2d(p),
                    nn.Dropout(dropout_visual)))
            in_chans = curr_chans
            
        self.visual_feature_extractor = nn.Sequential(*conv_layers_visual)
        self.visual_flatten = nn.Flatten(start_dim=2)
        self.visual_linear = nn.Linear(64,2400)
        # encoder setting
        self.encoder = nn.ModuleList()
        self.encoder_channel = [input_channel] + unet_channel

        # decoder setting
        self.decoder = nn.ModuleList()
        self.decoder_outchannel = unet_channel
        self.decoder_inchannel = list(map(lambda x:x[0] + x[1] ,zip(unet_channel[1:] + [256], unet_channel)))

        self.conv2d = nn.Conv2d(self.decoder_outchannel[0], input_channel, 1, 1)
        self.linear = nn.Linear(self.valid_freq * 2, self.valid_freq * 2)

        for idx in range(layer_number):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.encoder_channel[idx],
                        self.encoder_channel[idx+1],
                        self.kernel[idx],
                        self.stride[idx],
                    ),
                    nn.BatchNorm2d(self.encoder_channel[idx+1]),
                    nn.LeakyReLU(0.3)
                )
            )


        for idx in range(layer_number):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_inchannel[-1-idx],
                        self.decoder_outchannel[-1-idx],
                        self.kernel[-1-idx],
                        self.stride[-1-idx]
                    ),
                    nn.BatchNorm2d(self.decoder_outchannel[-1-idx]),
                    nn.LeakyReLU(0.3)
                )
            )
            
        

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
        features = features[:,:,:self.valid_freq,:,:]
        real_features = features[..., 0]
        imag_features = features[..., 1]

        return real_features, imag_features

    def encode_padding_size(self, kernel_size):
        k_f, k_t = kernel_size
        p_t_s = int(k_t / 2)
        p_f_s = int(k_f / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if k_t % 2 == 0:
            p_t_0 = p_t_0 - 1

        if k_f % 2 == 0:
            p_f_0 = p_f_0 - 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def decode_padding_size(self, in_size, target_size):
        i_f, i_t = in_size
        t_f, t_t = target_size
        p_t_s = int(abs(t_t - i_t) / 2)
        p_f_s = int(abs(t_f - i_f) / 2)

        p_t_0, p_t_1, p_f_0, p_f_1 = (p_t_s, p_t_s, p_f_s, p_f_s)

        if abs(t_t - i_t) % 2 == 1:
            p_t_1 = p_t_1 + 1

        if abs(t_f - i_f) % 2 == 1:
            p_f_1 = p_f_1 + 1

        return (p_t_0, p_t_1, p_f_0, p_f_1)

    def encode_padding_same(self, features, kernel_size):
        p_t_0, p_t_1, p_f_0, p_f_1 = self.encode_padding_size(kernel_size)

        features = F.pad(features, (p_t_0, p_t_1, p_f_0, p_f_1))

        return features

    def decode_padding_same(self, features, encoder_features, stride):
        # shape: [B, C, F, T]
        _, _, f, t = features.size()
        _, _, ef, et = encoder_features.size()

        # shape: [F, T]
        sf, st = stride
        tf, tt = (int(ef * sf), int(et * st))

        p_t_0, p_t_1, p_f_0, p_f_1 = self.decode_padding_size((f, t), (tf, tt))

        # shape: [B, C, F, T]
        if (p_t_0 != 0) or (p_t_1 != 0):
            features = features[:, :, :, p_t_0:-p_t_1]
        if (p_f_0 != 0) or (p_f_1 != 0):
            features = features[:, :, p_f_0:-p_f_1, :]

        return features

    def forward(self, inputs, images, device):
        # shape: [B, C, F, T]
        real_features, imag_features = self.extract_features(inputs, device)
        # shape: [B, C, F*2, T]
        features = torch.cat((real_features, imag_features), 2)

        visual_features = self.visual_feature_extractor(images)
        visual_features = self.visual_flatten(visual_features)
        visual_features = self.visual_linear(visual_features).view(inputs.shape[0], 256, 32, 75)

        out = features
        encoder_out = []
        # for idx in range(0, len(self.encoder), 2):
        #     out = self.encode_padding_same(out, self.kernel[idx//3])
        #     out = self.encoder[idx](out)
        #     # out = self.encoder[idx+1](out, visual_features)
        #     out = self.encoder[idx+1](out)
        #     encoder_out.append(out)
        #     print(f"Encoder {idx}: {out.shape}")
            
        for idx, layer in enumerate(self.encoder):
            out = self.encode_padding_same(out, self.kernel[idx])
            out = layer(out)
            encoder_out.append(out)
            # print(f"Encoder {idx}: {out.shape}")

        # print(encoder_out[-1].shape)
        # print(visual_features.shape)
        out = torch.cat((encoder_out[-1], visual_features), dim=1)
        
        for idx, layer in enumerate(self.decoder):
            if idx != 0:
                # print(out.shape)
                # print(encoder_out[-1-idx].shape)
                out = torch.cat((out, encoder_out[-1-idx]), 1)
            out = layer(out)
            # print(out.shape)
            out = self.decode_padding_same(out, encoder_out[-1-idx], self.stride[-1-idx])
            # print(out.shape, "\n")

        out = self.conv2d(out)
        # shape: [B, C, T, F*2]
        out = out.permute(0,1,3,2)
        out = self.linear(out)
        # shape: [B, C, F*2, T]
        out = out.permute(0,1,3,2)

        real_mask = out[:,:,:self.valid_freq,:]
        imag_mask = out[:,:,self.valid_freq:,:]

        est_speech_real = torch.mul(real_features, real_mask) - torch.mul(imag_features, imag_mask)
        est_speech_imag = torch.mul(real_features, imag_mask) + torch.mul(imag_features, real_mask)
        est_speech_stft = torch.complex(est_speech_real, est_speech_imag)

        # shape: [B, C, F, T]
        est_speech_stft = torch.sum(est_speech_stft, 1)
        batch_size, frequency, frame = est_speech_stft.size()
        est_speech_stft = torch.cat((est_speech_stft, torch.zeros(batch_size, 1, frame).to(device)), 1)
        print("Number of dimensions of est_speech_stft: ", est_speech_stft.ndim)
        print("Shape of dimensions of est_speech_stft:", est_speech_stft.shape)

        # shape: [B, S]
        est_speech = torch.istft(
                        est_speech_stft,
                        self.fft_size,
                        self.hop_size,
                        self.win_size,
                        torch.hann_window(self.win_size).to(device))
        # shape: [B, 1, S]
        print("Number of dimensions of est_speech: ", est_speech.ndim)
        print("Shape of dimensions of est_speech:", est_speech.shape)
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
    batch_size = 16
    audio_channel = 4
    length = int((frames_num - 1) * hop_size + fft_size - 4 * hop_size) # 4.792 seconds
    inputs = torch.rand(batch_size,audio_channel,length)

    model = MIMO_UNet_Beamforming()
    out = model(inputs, 'cpu')
    print('input size:', inputs.size())
    print('out size:', out.size())

    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

import os
import torch
import torch.utils.data as utils
import librosa
import numpy as np

import utility_functions as uf

from utility_functions import audio_image_csv_to_dict, load_image


class CustomAudioVisualDataset(utils.Dataset):
    def __init__(self, audio_predictors, audio_target, image_path=None, image_audio_csv_path=None,
                 transform_image=None):
        self.audio_predictors = audio_predictors[0]
        self.audio_target = audio_target
        self.audio_predictors_path = audio_predictors[1]
        self.image_path = image_path
        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_dict = audio_image_csv_to_dict(image_audio_csv_path)
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")

    def __len__(self):
        return len(self.audio_predictors)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_pred = self.audio_predictors[idx]
        audio_trg = self.audio_target[idx]
        audio_pred_path = self.audio_predictors_path[idx]

        if self.image_path:
            image_name = self.image_audio_dict[audio_pred_path]
            img = load_image(os.path.join(self.image_path, image_name))

            if self.transform:
                img = self.transform(img)

            return (audio_pred, img), audio_trg

        return audio_pred, audio_trg


# class CustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         transposed_data_0 = list(zip(*transposed_data[0]))
#         self.audio_pred = torch.stack(transposed_data_0[0], 0)
#         self.inp = list(zip(self.audio_pred, transposed_data_0[1]))
#         self.tgt = torch.stack(transposed_data[1], 0)

#     # custom memory pinning method on custom type
#     def pin_memory(self):
#         self.audio_pred = self.audio_pred.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self

# def collate_wrapper(batch):
#     return CustomBatch(batch)


'''
Questa versione light non richiede il precaricamento in memoria dei dati, ma procede a caricarli di volta in volta.
Maggior tempo richiesto a epoca, ma nessun vincolo su macchine con poca RAM.
Di fatto, integrazione di parte del preprocessing all'interno del custom dataset.
'''


class CustomLightAudioDataset(utils.Dataset):
    def __init__(self, folder, input_path, num_mics, segmentation_len=None, pad_length=4.792, train_val_split=0.7,
                 training=False, test=False, transform=None):
        super().__init__()

        self.folder = folder
        main_folder = os.path.join(input_path, folder)

        self.data_path = os.path.join(main_folder, 'data')

        data = os.listdir(self.data_path)
        self.data = [i for i in data if i.split('.')[0].split('_')[-1] == 'A']  # filter files with mic B

        self.sr_task1 = 16000
        self.max_file_length_task1 = 12

        self.num_mics = num_mics
        self.segmentation_len = segmentation_len
        self.pad_length = pad_length

        split_point = int(len(self.data) * train_val_split)
        if training:
            self.data = self.data[:split_point]
        elif not training and not test:
            self.data = self.data[split_point:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        def pad(x, size):
            # pad all sounds to 4.792 seconds to meet the needs of Task1 baseline model MMUB
            length = x.shape[-1]
            if length > size:
                pad = x[:, :size]
            else:
                pad = np.zeros((x.shape[0], size))
                pad[:, :length] = x
            return pad

        if torch.is_tensor(idx):
            idx = idx.tolist()

        predictors = []
        target = []

        sound = self.data[idx]

        sound_path = os.path.join(self.data_path, sound)
        target_path = '/'.join(
            (sound_path.split('/')[:-2] + ['labels'] + [sound_path.split('/')[-1]]))  # change data with labels
        target_path = target_path[:-6] + target_path[-4:]  # remove mic ID
        # target_path = sound_path.replace('data', 'labels').replace('_A', '')  #old wrong line
        samples, sr = librosa.load(sound_path, sr=self.sr_task1, mono=False)
        # image =
        # samples = pad(samples)
        if self.num_mics == 2:  # if both ambisonics mics are wanted
            # stack the additional 4 channels to get a (8, samples) shap
            B_sound_path = sound_path[:-5] + 'B' + sound_path[-4:]  # change A with B
            samples_B, sr = librosa.load(B_sound_path, sr=self.sr_task1, mono=False)
            samples = np.concatenate((samples, samples_B), axis=-2)

        samples_target, sr = librosa.load(target_path, sr=self.sr_task1, mono=False)
        samples_target = samples_target.reshape((1, samples_target.shape[0]))

        if self.segmentation_len is not None:
            # segment longer file to shorter frames
            # not padding if segmenting to avoid silence frames
            segmentation_len_samps = int(self.sr_task1 * self.segmentation_len)
            predictors_cuts, target_cuts = uf.segment_waveforms(samples, samples_target, segmentation_len_samps)
            for i in range(len(predictors_cuts)):
                predictors.append(torch.tensor(predictors_cuts[i]).float())
                target.append(torch.tensor(target_cuts[i]).float())

            return predictors, target
        else:
            samples = pad(samples, size=int(self.sr_task1 * self.pad_length))
            samples_target = pad(samples_target, size=int(self.sr_task1 * self.pad_length))
            predictor = torch.tensor(samples).float()
            target = torch.tensor(samples_target).float()
            return predictor, target
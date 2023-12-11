import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms

from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming, audiovisual_MIMO_UNet_Beamforming
from models.myAMMB import AMMB
from utility_functions import load_model, save_model
from custom_dataset import CustomAudioVisualDataset, CustomLightAudioDataset

from torch.optim.lr_scheduler import StepLR

import wandb

start_epoch = 0

WANDB = True

use_amp = True  # Settare a True se si vuole usare la automatic mixed precision

'''
Train our baseline model for the Task1 of the L3DAS23 challenge.
This script saves the model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task1.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''


def evaluate(model, device, criterion, dataloader):
    # compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            if "audiovisual" in args.architecture:
                audios = x[0].to(device)
                images = x[1].to(device)
                outputs = model(audios, images, device)
            else:
                x = x.to(device)
                outputs = model(x, device)
            loss = criterion(outputs, target)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current val loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss


def main(args):
    if args.use_cuda:
        device = 'cuda'  # + str(args.gpu_id)
    # elif args.use_mps:
    #     device = 'mps'
    else:
        device = 'cpu'

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # LOAD DATASET
    print('\nLoading dataset')

    # with open(args.training_predictors_path, 'rb') as f:
    #     training_audio_predictors = pickle.load(f)
    # with open(args.training_target_path, 'rb') as f:
    #     training_target = pickle.load(f)
    # with open(args.validation_predictors_path, 'rb') as f:
    #     validation_audio_predictors = pickle.load(f)
    # with open(args.validation_target_path, 'rb') as f:
    #     validation_target = pickle.load(f)
    # with open(args.test_predictors_path, 'rb') as f:
    #     test_predictors = pickle.load(f)
    # with open(args.test_target_path, 'rb') as f:
    #     test_target = pickle.load(f)

    # training_img_predictors = training_audio_predictors[1]
    # training_audio_predictors = np.array(training_audio_predictors[0])
    # training_target = np.array(training_target)
    # validation_img_predictors = validation_audio_predictors[1]
    # validation_audio_predictors = np.array(validation_audio_predictors[0])

    # # validation_img_predictors = validation_predictors[1] già comment.

    # validation_target = np.array(validation_target)
    # test_audio_predictors = np.array(test_predictors[0])
    # test_img_predictors = test_predictors[1]
    # test_target = np.array(test_target)

    # print ('\nShapes:')
    # print ('Training predictors: ', training_audio_predictors.shape)
    # print ('Validation predictors: ', validation_audio_predictors.shape)
    # print ('Test predictors: ', test_audio_predictors.shape)

    # #convert to tensor
    # training_audio_predictors = torch.tensor(training_audio_predictors).float()
    # validation_audio_predictors = torch.tensor(validation_audio_predictors).float()
    # test_audio_predictors = torch.tensor(test_audio_predictors).float()
    # training_target = torch.tensor(training_target).float()
    # validation_target = torch.tensor(validation_target).float()
    # test_target = torch.tensor(test_target).float()
    # #build dataset from tensors
    # # tr_dataset = utils.TensorDataset(training_predictors, training_target)
    # # val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    # # test_dataset = utils.TensorDataset(test_predictors, test_target)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Con vecchio preprocessing
    # tr_dataset = CustomAudioVisualDataset((training_audio_predictors, training_img_predictors), training_target, args.path_images, args.path_csv_images_train, transform)
    # val_dataset = CustomAudioVisualDataset((validation_audio_predictors,validation_img_predictors), validation_target, args.path_images, args.path_csv_images_train, transform)
    # test_dataset = CustomAudioVisualDataset((test_audio_predictors,test_img_predictors), test_target, args.path_images, args.path_csv_images_test, transform)

    # Utilizziamo il nuovo custom dataset Light
    tr_dataset = CustomLightAudioDataset('L3DAS22_Task1_train100', '/content/gdrive/MyDrive/project_folder/L3DAS23/DATASETS/Task1/', num_mics=1,
                                         train_val_split=0.7, training=True, transform=transform)
    val_dataset = CustomLightAudioDataset('L3DAS22_Task1_train100', '/content/gdrive/MyDrive/project_folder/L3DAS23/DATASETS/Task1/', num_mics=1,
                                          train_val_split=0.7, transform=transform)
    test_dataset = CustomLightAudioDataset('L3DAS22_Task1_dev', '/content/gdrive/MyDrive/project_folder/L3DAS23/DATASETS/Task1/',
                                           num_mics=1, test=True, transform=transform)

    # build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    # LOAD MODEL
    if args.architecture == 'fasnet':
        model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'tac':
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                           hidden_dim=args.hidden_dim, layer=args.layer,
                           segment_size=args.segment_size, nspk=args.nspk,
                           win_len=args.win_len, context_len=args.context_len,
                           sr=args.sr)
    elif args.architecture == 'MIMO_UNet_Beamforming':
        model = MIMO_UNet_Beamforming(fft_size=args.fft_size,
                                      hop_size=args.hop_size,
                                      input_channel=args.input_channel)
    elif args.architecture == 'audiovisual_MIMO_Unet_Beamforming':
        model = audiovisual_MIMO_UNet_Beamforming(fft_size=args.fft_size,
                                                  hop_size=args.hop_size,
                                                  input_channel=args.input_channel)
    elif args.architecture == 'AMMB':
        model = AMMB(4, 512, 3, 3, 4, 512, 128, 512, 0.1, 'cuda')
        # num_heads, embed_dim, num_encoders, global_B, channel_dim, fft_size, hop_size, win_size, dropout, device
    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    # compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total paramters: ' + str(model_params))

    # set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    # SCalare mixed precision
    # scheduler = StepLR(optimizer, step_size=10)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    # load model checkpoint if desired
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda)

    # TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    epoch = 1
    while state["worse_epochs"] < args.patience:
        print("Training epoch " + str(epoch))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    target = target.to(device)
                    if "audiovisual" in args.architecture:
                        audios = x[0].to(device)
                        images = x[1].to(device)
                    else:
                        x = x.to(device)
                    t = time.time()
                    # Compute loss for each instrument/model
                    # PRIMA c'era optimizer.zero_grad()
                    if "audiovisual" in args.architecture:
                        outputs = model(audios, images, device)
                    else:
                        outputs = model(x, device)
                    loss = criterion(outputs, target)

                scaler.scale(loss).backward()  # prima era loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                pbar.set_description("Current train loss: {:.4f}".format(train_loss))
                scaler.step(optimizer)  # prima era: optimizer.step()
                # scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            # PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion, val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            # valid_loss = val_loss.cpu().detach().numpy()
            # checkpoint_name = ('%03d' % epoch) + '_' + ('%.6f' % valid_loss) + '.pth'
            # checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint")
            checkpoint_epoca_path = os.path.join(args.checkpoint_epoca_dir, "checkpoint_epoca")


            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
                state["epoca_loss"] = val_loss
                state["epoca_checkpoint"] = checkpoint_epoca_path
                print("Saving not the best model...")
                save_model(model, optimizer, state, checkpoint_epoca_path)
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path
                state["epoca_checkpoint"] = checkpoint_epoca_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)
                save_model(model, optimizer, state, checkpoint_epoca_path)

            state["epochs"] += 1
            # state["worse_epochs"] = 200
            if WANDB:
                wandb.log({'epoch': state["epochs"], 'train_loss': train_loss, 'val_loss': val_loss})

            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())
            epoch += 1
    # LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], args.use_cuda)
    # compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion, tr_data)
    val_loss = evaluate(model, device, criterion, val_data)
    test_loss = evaluate(model, device, criterion, test_data)

    # PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print('RESULTS')
    for i in results:
        if 'hist' not in i:
            print(i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mnt_path = '/content/gdrive/MyDrive/project_folder/L3DAS23/'
    # saving parameters
    parser.add_argument('--results_path', type=str, default=mnt_path + 'RESULTS/Task1',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default=mnt_path + 'RESULTS/Task1',
                        help='Folder to write checkpoints into')
    parser.add_argument('--checkpoint_epoca_dir', type=str, default=mnt_path + 'RESULTS/Task1',
                        help='Folder to write epochs checkpoints into')
    parser.add_argument('--path_images', type=str, default=None,
                        help="Path to the folder containing all images of Task1. None when using the audio-only version")
    parser.add_argument('--path_csv_images_train', type=str,
                        default='DATASETS/Task1/L3DAS23_Task1_train/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo) in the train/val set")
    parser.add_argument('--path_csv_images_test', type=str, default='DATASETS/Task1/L3DAS23_Task1_dev/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo)")
    # dataset parameters
    parser.add_argument('--training_predictors_path', type=str,
                        default=mnt_path + 'DATASETS/processed/task1_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str,
                        default=mnt_path + 'DATASETS/processed/task1_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str,
                        default=mnt_path + 'DATASETS/processed/task1_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str,
                        default=mnt_path + 'DATASETS/processed/task1_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str,
                        default=mnt_path + 'DATASETS/processed/task1_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default=mnt_path + 'DATASETS/processed/task1_target_test.pkl')
    # training parameters
    # Modificato LR a 0.0005 (da 0.001) e batchsize raddoppiata a 12
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--use_mps', type=bool, default='False')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='True')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    # model parameters
    # Training includes images whenever 'audiovisual' appears in the architecture name
    parser.add_argument('--architecture', type=str, default='AMMB',
                        help="model name")
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=1)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--fft_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=4)

    args = parser.parse_args()

    if WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            #entity='thesis-i2i',
            project="AmbiSE_different_dim_L2",
            name="myAMMB",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.architecture,
                "dataset": "L3DAS22",
                "batch_size": args.batch_size,
                "patience": args.patience,
                "loss": args.loss,
                "enc_dim": args.enc_dim,
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "layer": args.layer,
                "segment_size": args.segment_size
            },

        )

    # eval string bools
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)

    main(args)

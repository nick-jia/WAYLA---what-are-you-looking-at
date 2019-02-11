import numpy as np
import argparse
import os
import gc
from time import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from utils import *
from dataset import SaliencyDataset
from model import *


# set random seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def load_data(x, y, batch_size):
    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = SaliencyDataset(x, y, data_transforms)
    loader = DataLoader(data, batch_size=batch_size)

    return loader


def evaluate(model, val_size, loss_fnc, bs, path, size, device):
    total_val_loss = 0.0
    i = 0
    while i < val_size:
        x, y = process_data(path, 'val', i, bs)
        val_loader = load_data(x, y, bs)

        for j, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs.to(device)).squeeze()
            loss = loss_fnc(outputs, labels.to(device).float().squeeze())
            total_val_loss += loss.item()

        del x, y
        gc.collect()
        i += size

    return float(total_val_loss) / (i // size + 1)


def main(arg):
    mGPU = 'False'
    model_name = 'model{}_seed{}.pt'.format(arg.model_num, seed)
    cur_epoch = 0
    best_loss = float('inf')
    save_label = 0

    if torch.cuda.is_available():
        print('Using GPU')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('Using CPU')
        torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_num = len([file for file in os.listdir(arg.path + '/fixations/train') if file.endswith('.mat')])
    if arg.val_size is None:
        val_num = len([file for file in os.listdir(arg.path + '/fixations/val') if file.endswith('.mat')])
    else:
        val_num = arg.val_size

    model = CombinedModel(arg.hidden_size, arg.kernel_size, arg.time_steps)

    if arg.resume == 'True':
        state = torch.load(model_name)
        cur_epoch += state['epoch']
        model.load_state_dict(state['state_dict'])
        print('Reusing model currently trained {} epochs'.format(cur_epoch))

    if torch.cuda.device_count() > 1:
        mGPU = 'True'
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # freeze parameters of RESNET-50
    if arg.freeze == 'True':
        print('Freezing ResNet')
        i = 0
        for param in model.parameters():
            if i == 159:  # freeze ResNet
                break
            param.requires_grad = False
            i += 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arg.lr, eps=1e-8)

    if arg.resume == 'True':
        optimizer.load_state_dict(state['optimizer'])
        print('Reusing optimizer')

    total_weights = arg.mse + arg.kldiv + arg.nss
    mse = arg.mse/total_weights
    kldiv = arg.kldiv/total_weights
    nss = arg.nss/total_weights
    loss_fnc = CombinedLoss(mse, kldiv, nss)  # can change weights
    print('start training')
    val_loss = evaluate(model, val_num, loss_fnc, arg.batch_size, arg.path, arg.data_size, device)
    print(' Loss at epoch {} is {}'.format(cur_epoch, val_loss))
    start_time = time()

    for epoch in range(arg.epochs - cur_epoch):  # loop over the dataset multiple times
        total_train_loss = 0.0
        i = 0
        while i < train_num:
            x, y = process_data(arg.path, 'train', i, arg.data_size)
            train_loader = load_data(x, y, arg.batch_size)

            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = loss_fnc(outputs, labels.to(device).float())
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                print('images {}, time passed is {}'.format(i, time() - start_time))
            del x, y
            gc.collect()
            i += arg.data_size
        train_loss = float(total_train_loss) / (i // arg.data_size + 1)
        val_loss = evaluate(model, val_num, loss_fnc, arg.batch_size, arg.path, arg.data_size, device)
        print("Epoch {}: Train loss: {} | Validation loss: {}".format(epoch + 1 + cur_epoch, train_loss, val_loss))
        print('{} seconds were used for epoch {}'.format(time() - start_time, epoch + 1 + cur_epoch))

        if arg.save == 'True':
            os.chdir(arg.path)
            if not os.path.exists(arg.path + '/' + model_name[:-3]):
                os.makedirs(model_name[:-3])
            os.chdir(arg.path + '/' + model_name[:-3])
            if save_label == 0:
                save_label = 1
                save_image(inputs[0, :, :, :], epoch + 1 + cur_epoch, 'label_1')
                save_image(inputs[1, :, :, :], epoch + 1 + cur_epoch, 'label_2')
                save_image(inputs[2, :, :, :], epoch + 1 + cur_epoch, 'label_3')
            save_saliency(outputs[0, :, :], epoch + 1 + cur_epoch, 'output_1')
            save_saliency(outputs[1, :, :], epoch + 1 + cur_epoch, 'output_2')
            save_saliency(outputs[2, :, :], epoch + 1 + cur_epoch, 'output_3')
            if val_loss < best_loss:
                best_loss = val_loss
                state = {'epoch': epoch + 1 + cur_epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, model_name)
                save_config(arg.model_num, best_loss, seed, arg, mGPU, cur_epoch)
            os.chdir(arg.path)
            print('Done saving')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data-size', type=int, default=128, help='number of data process every time, '
                                                                   'recommended to be the same as batch size')
    parser.add_argument('--val-size', type=int, default=2500, help='number of data used for validation, should '
                                                                   'be smaller than or equal to size of the '
                                                                   'validation dataset, which is 5000')
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--time-steps', type=int, default=5, help='number of time steps for running convLSTM')
    parser.add_argument('--save', type=str, default='True', help='whether save the model or not')
    parser.add_argument('--path', type=str, default=os.getcwd())
    parser.add_argument('--mse', type=int, default=1, help='weight of mse loss function')
    parser.add_argument('--kldiv', type=int, default=1, help='weight of kldiv loss function')
    parser.add_argument('--nss', type=int, default=1, help='weight of nss loss function')
    parser.add_argument('--resume', type=str, default='False', help='set to True if want to resume training a '
                                                                    'model that was trained before')
    parser.add_argument('--freeze', type=str, default='True', help='whether to freeze the parameters of RESNET '
                                                                   '50 or not')
    parser.add_argument('--model-num', type=int, default=1)

    arg = parser.parse_args()

    main(arg)

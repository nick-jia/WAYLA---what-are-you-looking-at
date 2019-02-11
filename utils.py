import numpy as np
import os
import json
from scipy.io import loadmat
from matplotlib import pyplot
from scipy import spatial


def process_data(data_path, category, starter, length):
    # Input: path to the data file, category (train/val), rank of starter image, and number of image to analyze.
    # Function: save the selected data to 2 npy files, data and label.
    os.chdir(data_path + '/fixations/{}'.format(category))
    file_list = [file for file in os.listdir('.') if file.endswith('.mat')]
    data = []
    label = []
    end = min(starter + length, len(file_list))

    for file in file_list[starter: end]:
        fix_map = np.zeros(loadmat(file)['resolution'][0][::-1])
        fixation = loadmat(file)['gaze']['fixations']
        fixation = np.vstack((i[0] for i in fixation))
        for point in fixation:
            fix_map[point[0] - 1, point[1] - 1] = 1
        label.append(fix_map.transpose())

        data.append(pyplot.imread(data_path + '/images/' + file[:-4] + '.jpg'))
    return np.array(data), np.array(label)


def fix_to_map(path):
    fix_map = np.zeros(loadmat(path)['resolution'][0][::-1])
    fixation = loadmat(path)['gaze']['fixations']
    fixation = np.vstack((i[0] for i in fixation))
    for point in fixation:
        fix_map[point[0] - 1, point[1] - 1] = 1
    pyplot.imsave(path[:-3] + 'png', fix_map.transpose(), cmap='gray')
    return fix_map.transpose()


def save_image(M, epoch, name):
    M = M.cpu().detach().numpy()
    # pyplot.imshow(M, cmap="gray")
    pyplot.imsave("output_epoch_{}_{}.png".format(epoch, name), M)


def save_saliency(M, epoch, name):
    M = M.cpu().detach().numpy()
    # pyplot.imshow(M, cmap="gray")
    pyplot.imsave("output_epoch_{}_{}.png".format(epoch, name), M, cmap="gray")


def save_config(model_num, loss, seed, args, mGPU, curr_epoch):
    data = {}
    data['batch_size'] = args.batch_size
    data['learning_rate'] = args.lr
    data['total epoch'] = args.epochs
    data['current epoch'] = curr_epoch
    data['data size'] = args.data_size
    data['validation size'] = args.val_size
    data['kernal size'] = args.kernel_size
    data['hidden size'] = args.hidden_size
    data['time steps'] = args.time_steps
    data['mse weight'] = args.mse
    data['kldiv weight'] = args.kldiv
    data['nss weight'] = args.nss
    data['freeze'] = args.freeze
    data['multiple GPU'] = mGPU

    with open('model{}_seed{}_loss{}.json'.format(model_num, seed, loss), 'w') as outfile:
        json.dump(data, outfile)


def cos_similarity(l1, l2):
    return 1 - spatial.distance.cosine(l1, l2)
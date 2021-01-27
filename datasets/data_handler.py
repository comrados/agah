import os
import numpy as np
import h5py
import scipy.io as scio


def preprocess(x, mean, std):
    mean, std = np.array(mean), np.array(std)
    return (x - mean.reshape(3, 1, 1)) / (std.reshape(3, 1, 1) + 1e-5)


def load_data(path, type='ucm'):
    print('Loading', type)
    if type == 'ucm':
        return load_ucm(path)
    elif type == 'flickr25k':
        return load_flickr25k(path)
    else:
        return load_nus_wide(path)


def load_flickr25k(path):
    data_file = h5py.File(path)
    images = data_file['images'][:]
    tags = data_file['YAll'][:]
    labels = data_file['LAll'][:]
    data_file.close()
    return images, tags, labels


def load_nus_wide(path_dir):
    image_file = h5py.File(os.path.join(path_dir, 'nus-wide-tc21-iall.mat'))
    images = image_file['IAll'][:]
    image_file.close()

    tag_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-yall.mat'))
    tags = tag_data['YAll']

    label_data = scio.loadmat(os.path.join(path_dir, 'nus-wide-tc21-lall.mat'))
    labels = label_data['LAll']

    return images, tags, labels


def data_enhance(images, tags, labels, c=0.5):
    num = images.shape[0]
    ind1 = np.random.permutation(num // 2)
    ind2 = np.random.permutation(num // 2)
    inhanced_imgs = c * images[ind1] + (1 - c) * images[ind2]
    inhanced_tags = c * tags[ind1] + (1 - c) * tags[ind2]
    inhanced_labels = c * labels[ind1] + (1 - c) * labels[ind2]
    new_imgs = np.concatenate((images, inhanced_imgs))
    new_tags = np.concatenate((tags, inhanced_tags))
    new_labels = np.concatenate((labels, inhanced_labels))
    return new_imgs, new_tags, new_labels


def load_ucm(path):
    with h5py.File(path, "r") as hf:
        images = hf['images'][:]
        images = (images - images.mean()) / images.std()
        labels = hf['labels'][:]
        tags = hf['bow'][:]
        images = duplicate_data(images, 5)  # 5 times more captions than images
        labels = duplicate_data(labels, 5)  # 5 times more captions than labels
    return images, tags, labels


def duplicate_data(data, n):
    """
    Duplicates each value of 0-dim n times
        for n = 3: (1, 2, 3) -> (1, 1, 1, 2, 2, 2, 3, 3, 3)

    :param data: original data
    :param n: number of duplications
    :return:
    """
    new_data = np.zeros((data.shape[0] * n, data.shape[1]), dtype=data.dtype)
    idx = 0
    for d in data:
        for i in range(n):
            new_data[idx] = d
            idx += 1
    return new_data


def load_pretrain_model(path):
    return scio.loadmat(path)

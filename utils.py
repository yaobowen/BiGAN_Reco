# All the util functions should be here
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
import math

def lrelu(x, alpha):
	return tf.maximum(alpha * x, x)

def load(path, dtype=np.float64):
    '''
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.
    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    Returns: A tuple of
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    '''
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
        	print ('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
            np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                # grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)
    whole = (class_names, X_train, y_train, X_val, y_val, X_test, y_test)
    dictdata = {"whole": whole}
    toyindex_train = generate_toy()
    toyindex_val =  np.random.choice(X_val.shape[0], 100)
    toy = (class_names, X_train[toyindex_train, ], y_train[toyindex_train], X_val[toyindex_val, ], y_val[toyindex_val], X_test, y_test)
    dictdata["toy"] = toy
    print("Whole data shape:\n")
    print("X_train: {}, y_train: {}, X_val: {}, y_val: {}, X_test: {}".format(whole[1].shape, whole[2].shape, whole[3].shape, whole[4].shape, whole[5].shape))
    print("Toy data shape:\n")
    print("X_train: {}, y_train: {}, X_val: {}, y_val: {}, X_test: {}".format(toy[1].shape, toy[2].shape, toy[3].shape, toy[4].shape, toy[5].shape))
    return dictdata

def load_data(path, prefix=""):
    X_train = np.load(os.path.join(path, prefix+"X_train.npy"))
    y_train = np.load(os.path.join(path, prefix+"y_train.npy"))
    X_val = np.load(os.path.join(path, prefix+"X_val.npy"))
    y_val = np.load(os.path.join(path, prefix+"y_val.npy"))
    X_test = np.load(os.path.join(path, prefix+"X_test.npy"))
    y_test = np.load(os.path.join(path, prefix+"y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_data(dictdata):
    np.save("class_names", dictdata["whole"][0])  
    np.save("X_train", np.transpose(dictdata["whole"][1], (0, 2, 3, 1)))
    np.save("y_train", dictdata["whole"][2])
    np.save("X_val", np.transpose(dictdata["whole"][3], (0, 2, 3, 1)))
    np.save("y_val", dictdata["whole"][4])
    np.save("X_test", np.transpose(dictdata["whole"][5], (0, 2, 3, 1)))
    np.save("y_test", dictdata["whole"][6])

    np.save("toy_class_names", dictdata["toy"][0])
    np.save("toy_X_train", np.transpose(dictdata["toy"][1], (0, 2, 3, 1)))
    np.save("toy_y_train", dictdata["toy"][2])
    np.save("toy_X_val", np.transpose(dictdata["toy"][3], (0, 2, 3, 1)))
    np.save("toy_y_val", dictdata["toy"][4])
    np.save("toy_X_test", np.transpose(dictdata["toy"][5], (0, 2, 3, 1)))
    np.save("toy_y_test", dictdata["toy"][6])

def getMiniBatch(X, Y = None, batch_size = 64):
    train_indicies = np.arange(X.shape[0])
    np.random.shuffle(train_indicies)
    for i in range(int(math.ceil(X.shape[0] / batch_size))):
    # generate indicies for the batch
        start_idx = (i * batch_size) % X.shape[0]
        idx = train_indicies[start_idx:start_idx+batch_size]
        if(Y is None):
            yield X[idx,:]
        else:
            yield X[idx,:], Y[idx]

def generate_toy(dtype=np.float64, size = 5):
    print("Generate Toy Data")
    arr = np.arange(200) * 500
    x = size - 1
    toyindex = np.repeat(arr, x + 1) + np.tile(np.arange(x + 1), arr.size)
    return toyindex

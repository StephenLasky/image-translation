import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import math

PATH_TO_DEOMPRESSED_CIFAR10 = "/Users/stephenlasky/Documents/CS6804/FinalProject/data/cifar-10-batches-py"
IM_SIZE = 32 * 32 * 3

# from cifar-10: used to open cifar 10 files
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# data_row is a ROW of data from the CIFAR data set. therefore, this only returns ONE image
def cifar_im_to_std_im(data_row, w, h):
    im = np.zeros((h, w, 3), dtype=np.uint8)
    offset = h * w

    pixel_idx = 0
    for row in range(0, h):
        for col in range(0, w):
            im[row, col, 0] = data_row[pixel_idx]                   # red channel
            im[row, col, 1] = data_row[pixel_idx + offset]          # blue channel
            im[row, col, 2] = data_row[pixel_idx + 2 * offset]      # green channel
            pixel_idx = pixel_idx + 1

    return im

def print_img(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()

# type can be either vector (3072x1) [0] or regular [1] (32x32x3)
def get_im_set(num, type=1):
    # first we need to get the files in this directory
    file_names = os.listdir(PATH_TO_DEOMPRESSED_CIFAR10)
    data_batch_names = []
    for file_name in file_names:
        if file_name.find("data_batch_") != -1:
            data_batch_names.append(file_name)
    data_batch_names = sorted(data_batch_names)

    print("Found {} data batch files".format(len(data_batch_names)))

    data = unpickle(PATH_TO_DEOMPRESSED_CIFAR10 + "/" + data_batch_names[0])
    data = data["data"]

    # put im set into an array
    ims = []
    if type == 0:
       for i in range(num):
           ims.append(data[i])
    elif type == 1:
        for i in range(num):
            im = cifar_im_to_std_im(data[i], 32, 32)
            ims.append(im)

    return ims

def vec_to_tensor(vec):
    t = []
    for i in range(len(vec)):
        t.append(float(vec[i]))
    t = torch.tensor(t)

    return t

def vecs_to_tensors(vecs):
    tens = []
    for vec in vecs:
        t = []
        for i in range(len(vec)):
            t.append(float(vec[i]))
        tens.append(t)
    tens = torch.tensor(tens)

    return tens

def tensor_to_vec(tns):
    v = []
    for i in range(len(tns)):
        x = tns[i]
        if x < 0.0:
            x = int(0)
        elif x >= 256.0:
            x = int(256)
        v.append(int(x))
    return v

def tensors_to_vecs(tens):
    vecs = []
    for i in range(len(tens)):
        vecs.append(tensor_to_vec(tens[i]))
    return vecs


def split_vec(vec):
    top_im = []
    bot_im = []

    f_size = len(vec)/3
    h_size = f_size / 2

    for j in range(3):
        c = j * f_size
        for i in range(h_size):
            top_im.append(vec[c+i])
            bot_im.append(vec[c+i+h_size])

    return (top_im, bot_im)

def comb_vec(xvec, yvec):
    assert len(xvec) == len(yvec)
    f_size = len(xvec) / 3

    vec = []

    # do red
    for i in range(f_size):
        vec.append(xvec[i])
    for i in range(f_size):
        vec.append(yvec[i])

    # do blue
    for i in range(f_size):
        vec.append(xvec[i + f_size])
    for i in range(f_size):
        vec.append(yvec[i + f_size])

    # do green
    for i in range(f_size):
        vec.append(xvec[i + 2 * f_size])
    for i in range(f_size):
        vec.append(yvec[i + 2 * f_size])

    return vec

# input: list of images that are stored in the height x width x channels matrix format
def combine_img(im_list):
    # first, compute the size of the canvas
    # we will make it the CEILING(sqrt(num_ims))
    num_im = len(im_list)
    canvas_size = int(math.ceil(math.sqrt(num_im)))

    # now, generate the canvas
    im_h = len(im_list[0])
    im_w = len(im_list[0][0,:])
    canvas_w = im_w  * canvas_size
    canvas_h = im_h * canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    im_num = 0
    for imy in range(0,canvas_size):
        for imx in range(0,canvas_size):
            canvas[im_h * imy : im_h * imy + im_h, im_w * imx : im_w * imx + im_w,] = im_list[im_num]
            im_num = im_num + 1
            if im_num >= num_im:
                break
        if im_num >= num_im:
            break

    return canvas
"""
Created on 2020/9/8

@author: Boyun Li
"""
import os

import cv2
import numpy as np
import torch
import random
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import init
from PIL import Image
import torch.nn.functional as F

class EdgeComputation(nn.Module):
    def __init__(self, test=False):
        super(EdgeComputation, self).__init__()
        self.test = test
    def forward(self, x):
        if self.test:
            x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

            # y = torch.Tensor(x.size()).cuda()
            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, :, 1:] += x_diffx
            y[:, :, :, :-1] += x_diffx
            y[:, :, 1:, :] += x_diffy
            y[:, :, :-1, :] += x_diffy
            y = torch.sum(y, 1, keepdim=True) / 3
            y /= 4
            return y
        else:
            x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])

            y = torch.Tensor(x.size())
            y.fill_(0)
            y[:, :, 1:] += x_diffx
            y[:, :, :-1] += x_diffx
            y[:, 1:, :] += x_diffy
            y[:, :-1, :] += x_diffy
            y = torch.sum(y, 0) / 3
            y /= 4
            return y.unsqueeze(0)


# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def unnormalize_image(normalized_image, means, stds):
    # Ensure the normalized_image has 3 channels (for RGB images)
    normalized_image = normalized_image.transpose(1, 2, 0)
    if normalized_image.shape[2] != 3:
        raise ValueError("Input normalized image should have 3 channels (RGB).")

    # Unnormalize each channel
    unnormalized_image = np.zeros_like(normalized_image, dtype=np.float32)
    for i in range(3):
        unnormalized_image[:, :, i] = (normalized_image[:, :, i] * stds[i]) + means[i]
    unnormalized_image *= 255
    # Clip values to be within valid image range (0 to 255)
    unnormalized_image = np.clip(unnormalized_image, 0, 255).astype(np.uint8)

    return unnormalized_image

def save_feature_vec(img, num=0, save_dir='features/', gap=False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # img = F.interpolate(img, (64, 64), mode="bilinear")

    img = img.detach().cpu().numpy()

    if gap:
        img = np.mean(img, axis=(-2, -1))
    else:
        img = np.mean(img, axis=1)
        img = img[0].reshape(-1)

    np.save(save_dir + str(num) + '.npy', img)


def save_dce_features_full(img, num=0, save_dir='attn_maps/', img_size=None):
    H, W = img_size[2:]
    W = W * 2
    Bc, L, D = img.size()
    degrad_context = img[:, :L // 2, :][:, 1:].detach().cpu().numpy()
    clean_context = img[:, L // 2:, :][:, 1:].detach().cpu().numpy()

    for idx in range(Bc):
        for j in range(D):
            degrad_context_i = np.asarray(degrad_context[idx, :, j])
            degrad_context_i = degrad_context_i.reshape(7, 7)
            clean_context_i = np.asarray(clean_context[idx, :, j])
            clean_context_i = clean_context_i.reshape(7, 7)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.subplot(1, 1, 1)
            #  plt.imshow(degrad_context_i, cmap='viridis')
            plt.imshow(degrad_context_i)
            plt.axis('off')

            plt.savefig(save_dir + str(num) + '_' + str(j) + 'degrad.jpg', bbox_inches='tight',
                        pad_inches=0)
            plt.clf()  # Clear the figure for the next iteration
            image = Image.open(save_dir + str(num) + '_' + str(j) + 'degrad.jpg')
            image = image.resize(img_size[2:][::-1])
            image.save(save_dir + str(num) + '_' + str(j) + 'degrad.jpg')

            plt.subplot(1, 1, 1)
            #plt.imshow(clean_context_i, cmap='viridis')
            plt.imshow(clean_context_i)
            plt.axis('off')

            plt.savefig(save_dir + str(num) + '_' + str(j) + 'clean.jpg', bbox_inches='tight',
                        pad_inches=0)
            plt.clf()  # Clear the figure for the next iteration
            image = Image.open(save_dir + str(num) + '_' + str(j) + 'clean.jpg')
            image = image.resize(img_size[2:][::-1])
            image.save(save_dir + str(num) + '_' + str(j) + 'clean.jpg')


def save_dce_features(img, num=0, save_dir='attn_maps/', img_size=None):
    H, W = img_size[2:]
    W = W * 2
    Bc, L, D = img.size()
    degrad_context = img[:, :L//2, :][:, 1:]
    clean_context = img[:, L//2:, :][:, 1:]

    degrad_context = degrad_context.mean(dim=2)
    clean_context = clean_context.mean(dim=2)

    degrad_context_img = degrad_context.reshape(7, 7)
    clean_context_img = clean_context.reshape(7, 7)
    degrad_context_img = degrad_context_img.detach().cpu().numpy()
    clean_context_img = clean_context_img.detach().cpu().numpy()
    # final_dce = torch.cat([degrad_context_img, clean_context_img], dim=0)
    # final_dce = final_dce.detach().cpu().numpy()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.subplot(1, 1, 1)
    plt.imshow(degrad_context_img, cmap='viridis')
    plt.axis('off')

    plt.savefig(save_dir + str(num) + 'degrad_mean.jpg', bbox_inches='tight',
                pad_inches=0)
    plt.clf()  # Clear the figure for the next iteration
    image = Image.open(save_dir + str(num) + 'degrad_mean.jpg')
    image = image.resize(img_size[2:][::-1])
    image.save(save_dir + str(num) + 'degrad_mean.jpg')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.subplot(1, 1, 1)
    plt.imshow(clean_context_img, cmap='viridis')
    plt.axis('off')

    plt.savefig(save_dir + str(num) + 'clean_mean.jpg', bbox_inches='tight',
                pad_inches=0)
    plt.clf()  # Clear the figure for the next iteration
    image = Image.open(save_dir + str(num) + 'clean_mean.jpg')
    image = image.resize(img_size[2:][::-1])
    image.save(save_dir + str(num) + 'clean_mean.jpg')


def save_actual_attention(map, save_dir='attention/', img_size=None):
    B, nH, H, W = map.size()
    map = map[0].mean(dim=0).detach().cpu().numpy()

    result_array = np.delete(map, [0, 50], axis=0)  # Remove the 0th and 50th rows
    result_array = np.delete(result_array, [0, 50], axis=1)  # Remove the 0th and 50th columns

    truncated = result_array[:49, :]

    truncated = (truncated - np.min(truncated) / (np.max(truncated) - np.min(truncated)))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.subplot(1, 1, 1)
    plt.imshow(truncated, cmap='viridis')
    plt.axis('off')

    plt.savefig(save_dir + 'mean.jpg', bbox_inches='tight',
                pad_inches=0)
    plt.clf()  # Clear the figure for the next iteration
    image = Image.open(save_dir + 'mean.jpg')
    image = image.resize(img_size[2:][::-1])
    image.save(save_dir + 'mean.jpg')



def save_attn_maps(img, num=0, save_dir='attn_maps/', img_size=None):
    B, Cc, Hw, W = img.size()

    # img = img.reshape(B, Cc, 10, 10)
    img = img.detach().cpu().numpy()

    mean = np.mean(img[0], axis=0)
    mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))

    degrad = np.asarray(mean)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.subplot(1, 1, 1)
    plt.imshow(degrad, cmap='viridis')
    plt.axis('off')

    plt.savefig(save_dir + str(num) + 'mean.jpg', bbox_inches='tight',
                pad_inches=0)
    plt.clf()  # Clear the figure for the next iteration
    image = Image.open(save_dir + str(num) + 'mean.jpg')
    image = image.resize(img_size[2:][::-1])
    image.save(save_dir + str(num) + 'mean.jpg')

    for idx in range(B):
        for j in range(Cc):
            degrad = np.asarray(img[idx, j])

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.subplot(1, 1, 1)
            plt.imshow(degrad, cmap='viridis')
            plt.axis('off')

            plt.savefig(save_dir + str(num) + '_' + str(j) + '.jpg', bbox_inches='tight',
                        pad_inches=0)
            plt.clf()  # Clear the figure for the next iteration
            image = Image.open(save_dir + str(num) + '_' + str(j) + '.jpg')
            image = image.resize(img_size[2:][::-1])
            image.save(save_dir + str(num) + '_' + str(j) + '.jpg')

def save_attn_maps_(img, num=0, save_dir='attn_maps/', img_size=None):
    B, Cc, Hw, W = img.size()

    # img = img.reshape(B, Cc, 10, 10)
    img = img.detach().cpu().numpy()

    mean = np.mean(img[0], axis=0)
    mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))

    degrad = np.asarray(mean)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.subplot(1, 1, 1)
    plt.imshow(degrad, cmap='viridis')
    plt.axis('off')

    plt.savefig(save_dir + str(num) + '.jpg', bbox_inches='tight',
                pad_inches=0)
    plt.clf()  # Clear the figure for the next iteration
    image = Image.open(save_dir + str(num) + '.jpg')
    image = image.resize(img_size[2:][::-1])
    image.save(save_dir + str(num) + '.jpg')
    return

def save_forward_imgs(img, num, context_pair, save_dir='results_input/'):
    B, _, _, _ = img.size()
    img = img.detach().cpu().numpy()
    dctx, cctx = context_pair
    dctx = dctx.detach().cpu().numpy()
    cctx = cctx.detach().cpu().numpy()

    means = [0.48145466, 0.4578275, 0.40821073]
    stds = [0.26862954, 0.26130258, 0.27577711]
    for idx in range(B):
        img_norm = unnormalize_image(np.asarray(dctx[idx]), means, stds)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_dctx.png', img_norm)

    for idx in range(B):
        img_norm = unnormalize_image(np.asarray(cctx[idx]), means, stds)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_cctx.png', img_norm)

    for idx in range(B):
        degrad = np.asarray(img[idx]) * 255
        degrad = degrad.transpose(1, 2, 0)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        degrad = degrad.astype(np.uint8)
        degrad = cv2.cvtColor(degrad, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_img.png', degrad)

def debug_images(img, num, save_dir='results_input/', mode='normal'):
    B, _, _, _ = img.size()
    img = img.detach().cpu().numpy()
    if mode == 'clip':
        means = [0.48145466, 0.4578275, 0.40821073]
        stds = [0.26862954, 0.26130258, 0.27577711]
        for idx in range(B):
            img_norm = unnormalize_image(np.asarray(img[idx]), means, stds)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_img.png', img_norm)

    elif mode == 'normal':
        for idx in range(B):
            degrad = np.asarray(img[idx]) * 255
            degrad = degrad.transpose(1, 2, 0)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            degrad = degrad.astype(np.uint8)

            cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_img.png', degrad)
def resize_img(image, idx, max_size=512):
    h, w, _ = image.shape
    new_h = h
    new_w = w
    if h > max_size:
        new_h = 512
    if w > max_size:
        new_w = 512
    if new_h != h or new_w != w:
        image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        # print("old shape", h, w, idx)
        # print("new shape", image.shape, idx)
    return image


# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


# patches (B, H, W, C) -> image (H, W, C)
def splice_patches2image(patches, image_size, overlap=0):
    assert len(image_size) > 1
    assert patches.shape[-3] == patches.shape[-2]
    H = image_size[0]
    W = image_size[1]
    patch_size = patches.shape[-2] - overlap
    image = np.zeros(image_size)
    idx = 0
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            image[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = patches[idx,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 overlap:patch_size + overlap,
                                                                                                 :]
            idx += 1
    return image


# def data_augmentation(image, mode):
#     if mode == 0:
#         # original
#         out = image.numpy()
#     elif mode == 1:
#         # flip up and down
#         out = np.flipud(image)
#     elif mode == 2:
#         # rotate counterwise 90 degree
#         out = np.rot90(image, axes=(1, 2))
#     elif mode == 3:
#         # rotate 90 degree and flip up and down
#         out = np.rot90(image, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 4:
#         # rotate 180 degree
#         out = np.rot90(image, k=2, axes=(1, 2))
#     elif mode == 5:
#         # rotate 180 degree and flip
#         out = np.rot90(image, k=2, axes=(1, 2))
#         out = np.flipud(out)
#     elif mode == 6:
#         # rotate 270 degree
#         out = np.rot90(image, k=3, axes=(1, 2))
#     elif mode == 7:
#         # rotate 270 degree and flip
#         out = np.rot90(image, k=3, axes=(1, 2))
#         out = np.flipud(out)
#     else:
#         raise Exception('Invalid choice of image transformation')
#     return out

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


# def random_augmentation(*args):
#     out = []
#     if random.randint(0, 1) == 1:
#         flag_aug = random.randint(1, 7)
#         for data in args:
#             out.append(data_augmentation(data, flag_aug).copy())
#     else:
#         for data in args:
#             out.append(data)
#     return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)
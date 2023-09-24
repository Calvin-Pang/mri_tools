import numpy as np
import torch

def kspace_zero_padding_resample(input, ratio, range=255):
    '''
        in1-input: a numpy array
        in2-ratio: if ratio > 1, then it is unsampling which is completed by zero padding
                   if ratio < 1: then it is downsampling which is completed by k-space center crop
        output: a numpy array
    '''
    h, w = input.shape[-2:]
    h_new, w_new = round(ratio * h), round(ratio * w)
    f = np.fft.fftshift(np.fft.fft2(input))
    dh, dw = ((h_new - h) // 2, (w_new - w) // 2)
    if dh >= 0 and dw >= 0:
        f_resized = np.pad(f, ((dh, dh), (dw, dw)), mode='constant')
    else:
        f_resized = f[-dh: h + dh, -dw: w + dw]
    img_ds = np.fft.ifft2(np.fft.ifftshift(f_resized))
    img_ds = np.real(img_ds)
    img_ds = (img_ds - np.min(img_ds)) / (np.max(img_ds) - np.min(img_ds))
    return np.uint8(img_ds*range)

def kspace_zero_padding_resample_torch(input, ratio):
    '''
        in1-input: a torch tensor
        in2-ratio: if ratio > 1, then it is unsampling which is completed by zero padding
                   if ratio < 1: then it is downsampling which is completed by k-space center crop
        output: a torch tensor
    '''
    h, w = input.shape[-2:]
    h_new, w_new = round(ratio * h), round(ratio * w)
    f = torch.fft.fftshift(np.fft.fft2(input))
    dh, dw = ((h_new - h) // 2, (w_new - w) // 2)
    if dh >= 0 and dw >= 0:
        f_resized = torch.pad(f, ((dh, dh), (dw, dw)), mode='constant')
    else:
        f_resized = f[-dh: h + dh, -dw: w + dw]
    img_ds = torch.fft.ifft2(np.fft.ifftshift(f_resized))
    img_ds = torch.real(img_ds)
    img_ds = (img_ds - torch.min(img_ds)) / (torch.max(img_ds) - torch.min(img_ds))
    return img_ds
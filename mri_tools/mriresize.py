import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
def kspace_zero_padding_resample(input, ratio):
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
    info = np.iinfo(input.dtype)
    if dh >= 0 and dw >= 0:
        f_resized = np.zeros((h_new, w_new), dtype=np.complex64)
        f_resized[dh:dh+h, dw:dw+w] = f
        img_ds = np.fft.ifft2(np.fft.ifftshift(f_resized))
        img_ds = np.real(img_ds).clip(info.min,info.max)
    else:
        f_resized = np.zeros((h, w), dtype=np.complex64)
        f_resized[(h//2 - h_new//2):(h//2 + h_new//2), 
                  (w//2 - w_new//2):(w//2 + w_new//2)] = f[(h//2 - h_new//2):(h//2 + h_new//2), 
                                                           (w//2 - w_new//2):(w//2 + w_new//2)]
        img_ds = np.fft.ifft2(np.fft.ifftshift(f_resized))
        img_ds = np.real(img_ds).clip(info.min,info.max) 
        img_ds = cv2.resize(img_ds, (h_new, w_new), cv2.INTER_CUBIC) 
    img_ds = img_ds.astype(input.dtype)
    return img_ds


def kspace_zero_padding_resample_torch(input, ratio):
    '''
        in1-input: a torch tensor
        in2-ratio: if ratio > 1, then it is unsampling which is completed by zero padding
                   if ratio < 1: then it is downsampling which is completed by k-space center crop
        output: a torch tensor
        all data here is in range 0-1
    '''
    h, w = input.shape[-2:]
    h_new, w_new = round(ratio * h), round(ratio * w)
    f = torch.fft.fftshift(torch.fft.fft2(input))
    dh, dw = ((h_new - h) // 2, (w_new - w) // 2)
    if dh >= 0 and dw >= 0:
        f_resized = torch.zeros((1,h_new, w_new), dtype=torch.complex64, device=input.device)
        f_resized[:,dh:dh+h, dw:dw+w] = f
        img_ds = torch.fft.ifft2(torch.fft.ifftshift(f_resized))
        img_ds = torch.real(img_ds).clamp(0,1)
    else:
        f_resized = torch.zeros((1,h, w), dtype=torch.complex64, device=input.device)
        f_resized[:,(h//2 - h_new//2):(h//2 + h_new//2), 
                  (w//2 - w_new//2):(w//2 + w_new//2)] = f[:,(h//2 - h_new//2):(h//2 + h_new//2), 
                                                           (w//2 - w_new//2):(w//2 + w_new//2)]
        img_ds = torch.fft.ifft2(torch.fft.ifftshift(f_resized))
        img_ds = torch.real(img_ds).clamp(0,1) 
        img_ds = transforms.ToTensor()(transforms.Resize((h_new, w_new), 
                                                         Image.Resampling.BICUBIC)(transforms.ToPILImage()(img_ds)))      
    return img_ds

import numpy as np
from tqdm import tqdm

def norm_array(x : np.ndarray, phots : np.float32):
    #x must be a 3d array of shape (n_digits, pixels, pixels)
    x_norm = x.sum(axis = (1, 2))
    return phots*x/x_norm.reshape(x_norm.shape[0], 1, 1)

def generate_noisy_samples(mean : np.ndarray, phots : np.float32, n_frames : np.uint16, SNR : np.float32):
    #mean must be a 2d array of shape (pixels, pixels)
    dark_phots = phots/SNR
    pixels = mean.shape[-1]
    frames = np.random.poisson(lam = mean, size = (n_frames, pixels, pixels)) + \
             np.random.poisson(lam = np.ones((pixels, pixels))*dark_phots/(pixels**2), \
                               size = (n_frames, pixels, pixels))
    
    return frames
    
def create_noisy_dataset(images : np.ndarray, phots : np.float32, n_frames : np.uint16, SNR : np.float32):
    #images must be a 3d array of shape (n_digits, pixels, pixels)
    image_array = norm_array(images, phots = phots)
    n_digits = image_array.shape[0]
    #
    noisy_frames = []
    for i in tqdm(range(n_digits)):
        noisy_frames.append(generate_noisy_samples(mean = image_array[i], phots = phots, \
                                                   n_frames = n_frames, SNR = SNR).astype(np.uint8))
    noisy_frames = np.array(noisy_frames).astype(np.uint8)
    return noisy_frames
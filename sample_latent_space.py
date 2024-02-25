#!/usr/bin/env python3

import argparse, os, sys
import yaml

import numpy as np
import matplotlib.pyplot as plt

from src.abstract_cvae import AbstractCVAE
from src.load_model import load_model_from_directory

import tensorflow as tf
import tensorflow_datasets as tfds


gpu_list = tf.config.list_physical_devices('GPU')
# Calling GPUs by default with Keras will reserve the rest of the remaining memory
# To avoid this, allow memory growth to dynamically allocate memory over the program life
if gpu_list:
    try:
        for gpu in gpu_list:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f'TensorFlow Version: {tf.__version__}')
print(f'Num of GPUs: {len(tf.config.list_physical_devices("GPU"))}')


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Log filepath directory')
    parser.add_argument('--output-filename', '-o', type=str, default='latent_sample.png')
    parser.add_argument('--min-z', type=float, default=-1.0)
    parser.add_argument('--max-z', type=float, default=+1.0)
    return parser.parse_args()


def sample_latent_space(config:dict, model: AbstractCVAE, output_filename: str, min_z: float, max_z: float, N:int=10):

    latent_dim = config['model']['latent_dimensions']
    sample_size = (N*N, latent_dim)

    print(f'Sample Size: {sample_size}')

    z_sample = np.random.random(size=(N*N, latent_dim)) * (max_z - min_z) + min_z

    x_r = model.decode(z_sample, True)

    print(f'Reconstruction Size: {x_r.shape}')

    fig, ax_mat = plt.subplots(N, N)

    for row in range(N):
        for col in range(N):
            idx = row*N + col
            ax_mat[row][col].imshow(x_r[idx,:,:,:])
            ax_mat[row][col].axis('off')

    fig.savefig(output_filename, bbox_inches='tight')


def main():

    args = get_args()
    model, config = load_model_from_directory(args.log_dir)

    sample_latent_space(config, model, args.output_filename, args.min_z, args.max_z)


if __name__ == '__main__':
    main()


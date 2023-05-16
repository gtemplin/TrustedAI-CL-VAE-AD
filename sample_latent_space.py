#!/usr/bin/env python3

import argparse, os, sys
import yaml

import numpy as np
import matplotlib.pyplot as plt

from src.fuzzy_vae import FuzzyVAE

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


def load_config(config_filename: str):

    assert(os.path.exists(config_filename))
    assert(os.path.isfile(config_filename))

    # Load config file
    config = None
    try:
        with open(config_filename, 'r') as ifile:
            config = yaml.safe_load(ifile)

    except IOError as e:
        raise e
    except yaml.YAMLError as e:
        raise e

    return config 


def get_model_config(log_dir:str) -> tuple:

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    config_path = os.path.join(log_dir, 'config.yml')
    config = load_config(config_path)

    model = FuzzyVAE(config)
    model.load_model(log_dir)

    return model, config


def sample_latent_space(config:dict, model: FuzzyVAE, output_filename: str, min_z: float, max_z: float, N:int=10):

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
    model, config = get_model_config(args.log_dir)

    sample_latent_space(config, model, args.output_filename, args.min_z, args.max_z)


if __name__ == '__main__':
    main()


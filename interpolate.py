#!/usr/bin/env python3

import argparse, os, sys
import yaml
import time

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
    parser.add_argument('log_dir', type=str, help='Log Directory containing model and config')
    parser.add_argument('--sample-points', '-k', type=int, default=10, help='Number of samples to walk')
    parser.add_argument('--output-path', '-o', type=str, default='interpolate_output.png')
    return parser.parse_args()
    

def example_interpolate(config: dict, model: AbstractCVAE, output_path: str, k_sample_points:int=10):

    N = 10

    tf.random.set_seed(42)

    dataset_path = config['data'].get('dataset_path')
    dataset_name = config['data'].get('dataset')
    train_split = config['data']['train_split']
    config_img_size = config['data']['image_size']
    img_size = (config_img_size[0], config_img_size[1])

    if dataset_path is not None:
        print(f'Loading dataset from: {dataset_path}')
        assert(os.path.exists(dataset_path))
        assert(os.path.isdir(dataset_path))

        train_ds = tf.data.Dataset.load(os.path.join(dataset_path, 'train'))
        val_ds = tf.data.Dataset.load(os.path.join(dataset_path, 'validation'))

        def normalize_img(element):
            return tf.cast(element['image'], tf.float32) / 255.
        
        data = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        data = tfds.load(dataset_name, split=train_split, shuffle_files=False)

        def normalize_img(element):
            return tf.cast(element['image'], tf.float32) / 255.
        
        data = data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    def resize_img(element, img_size):
        return tf.image.resize(element, size=img_size)
    
    data = data.map(lambda x: resize_img(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    data = tf.convert_to_tensor([d for d in data.take(N*2)])

    t_vec = np.arange(k_sample_points)

    _, zvec, _, _ = model.call_detailed(data)

    z0 = zvec[:N]
    z1 = zvec[N:]

    z_delta = (z1 - z0) / k_sample_points

    r_vec = list()
    for t in t_vec:
        z = tf.reshape(z_delta * t + z0, shape=(N, -1))
        r_vec.append(model.decode(z, True))

    r_vec = tf.convert_to_tensor([data[:N]] + r_vec + [data[N:]])

    fig, ax_vec = plt.subplots(N, len(r_vec))

    for row in range(N):
        for col in range(len(r_vec)):
            ax_vec[row][col].imshow(r_vec[col,row,:,:,:])
            ax_vec[row][col].axis('off')

    title_font_size=8

    ax_vec[0][0].set_title('X0', fontsize=title_font_size)
    ax_vec[0][-1].set_title('X1', fontsize=title_font_size)

    for i in range(len(ax_vec[0])-2):
        ax_vec[0][i+1].set_title(f't{i}', fontsize=title_font_size)

    fig.savefig(output_path, bbox_inches='tight')



def main():

    args = get_args()
    model, config = load_model_from_directory(args.log_dir)
    example_interpolate(config, model, args.output_path, args.sample_points)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse, os, sys

import numpy as np
import matplotlib.pyplot as plt

from train import load_config
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
    parser.add_argument('log_dir', type=str, help='Log Directory containing model and config')
    parser.add_argument('--sample-points', '-k', type=int, default=10, help='Number of samples to walk')
    parser.add_argument('--output-path', '-o', type=str, default='interpolate_output.png')
    return parser.parse_args()

def load_model(log_dir: str):

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    config_path = os.path.join(log_dir, 'config.yml')
    config = load_config(config_path)

    model = FuzzyVAE(config)
    model.load_model(log_dir)

    return model, config
    

def example_interpolate(config: dict, model: FuzzyVAE, output_path: str, k_sample_points:int=10):

    dataset_name = config['data']['dataset']
    val_split = config['data']['val_split']
    config_img_size = config['data']['image_size']
    img_size = (config_img_size[0], config_img_size[1])

    data = tfds.load(dataset_name, split=val_split, shuffle_files=True)

    def normalize_img(element):
        return tf.cast(element['image'], tf.float32) / 255.
    
    def resize_img(element, img_size):
        return tf.image.resize(element, size=img_size)
    
    data = data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(lambda x: resize_img(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    data = tf.convert_to_tensor([d for d in data.take(2)])

    t_vec = np.arange(k_sample_points+1)

    _, zvec, _, _ = model.call_detailed(data)
    #_, z1, _, _ = model.call_detailed(data[1])

    z0 = zvec[0]
    z1 = zvec[1]

    z_delta = (z1 - z0) / k_sample_points

    r_vec = list()
    for t in t_vec:
        z = tf.reshape(z_delta * t + z0, shape=(1, -1))
        r_vec.append(model.decode(z, True)[0])

    fig, ax_vec = plt.subplots(1, len(r_vec)+2)
    r_vec = [data[0]] + r_vec + [data[1]]

    for img, ax in zip(r_vec, ax_vec):
        ax.imshow(img)
        ax.axis('off')
        print(np.min(img), np.max(img))

    fig.savefig(output_path, bbox_inches='tight')



def main():

    args = get_args()
    model, config = load_model(args.log_dir)
    example_interpolate(config, model, args.output_path, args.sample_points)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse, os, sys
import yaml
import time

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
    parser.add_argument('log_dir', type=str, help='Log Directory containing model and config')
    parser.add_argument('--sample-points', '-k', type=int, default=10, help='Number of samples to walk')
    parser.add_argument('--output-path', '-o', type=str, default='interpolate_output.png')
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


def load_model(log_dir: str):

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    config_path = os.path.join(log_dir, 'config.yml')
    config = load_config(config_path)

    model = FuzzyVAE(config)
    model.load_model(log_dir)

    return model, config
    

def example_interpolate(config: dict, model: FuzzyVAE, output_path: str, k_sample_points:int=10):

    N = 10

    tf.random.set_seed(42)

    dataset_name = config['data']['dataset']
    val_split = config['data']['train_split']
    config_img_size = config['data']['image_size']
    img_size = (config_img_size[0], config_img_size[1])

    data = tfds.load(dataset_name, split=val_split, shuffle_files=False)

    def normalize_img(element):
        return tf.cast(element['image'], tf.float32) / 255.
    
    def resize_img(element, img_size):
        return tf.image.resize(element, size=img_size)
    
    data = data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
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
    model, config = load_model(args.log_dir)
    example_interpolate(config, model, args.output_path, args.sample_points)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse, os, sys
import yaml

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.fuzzy_vae import FuzzyVAE

import tensorflow as tf

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
    parser.add_argument('log_dir', type=str)
    parser.add_argument('img_a', type=str, help='Source Image')
    parser.add_argument('img_b', type=str, help='Attribute 1')
    parser.add_argument('img_c', type=str, help='Attribute 2')
    parser.add_argument('--output-filename', '-o', type=str, default='j_diagram.png')

    return parser.parse_args()

def load_images(args, config:dict):

    print('Loading Images')

    img_a_filename = args.img_a
    img_b_filename = args.img_b
    img_c_filename = args.img_c

    assert(os.path.exists(img_a_filename))
    assert(os.path.isfile(img_a_filename))
    assert(os.path.exists(img_b_filename))
    assert(os.path.isfile(img_b_filename))
    assert(os.path.exists(img_c_filename))
    assert(os.path.isfile(img_c_filename))

    img_size = config['data']['image_size']
    img_size = (img_size[0], img_size[1])

    print(f'Loading A (Source) Image: {img_a_filename}')
    img_a = np.asarray(Image.open(img_a_filename))
    print(f'Loading B (Attr. 1) Image: {img_b_filename}')
    img_b = np.asarray(Image.open(img_b_filename))
    print(f'Loading C (Attr. 2) Image: {img_c_filename}')
    img_c = np.asarray(Image.open(img_c_filename))

    img_a = tf.image.resize(tf.cast(img_a, tf.float32) / 255., size=img_size)
    img_b = tf.image.resize(tf.cast(img_b, tf.float32) / 255., size=img_size)
    img_c = tf.image.resize(tf.cast(img_c, tf.float32) / 255., size=img_size)

    return [img_a, img_b, img_c]


def load_config(config_filename: str):

    print(f'Loading config from: {config_filename}')

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

    print(f'Loading model from: {log_dir}')

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    config_path = os.path.join(log_dir, 'config.yml')
    config = load_config(config_path)

    model = FuzzyVAE(config)
    model.load_model(log_dir)

    return model, config


def plot_j_diagram(model: FuzzyVAE, imgs: list, output_filename: str, N:int=10):

    # Get z space
    x_hat, z, _, _ = model.call_detailed(tf.convert_to_tensor(imgs), False)

    # Calculate Attribute Vectors
    z0 = z[0]           # A (Source)
    zba = z[1] - z[0]   # Attribute 1 Vector
    zca = z[2] - z[0]   # Attribute 2 Vector

    # Get Cartesian t-sample coordinates
    t0_vec = np.linspace(0.0, 1.0, N)
    t1_vec = np.linspace(0.0, 1.0, N)

    # Generate J-Diagram Subplot Grid
    fig, ax = plt.subplots(N+1, N+1)
    fig.suptitle('J-Diagram')

    # Set Source Image
    ax[0][0].imshow(imgs[0])
    ax[0][0].axis('off')
    ax[0][0].set_title('Source')

    # Set Attribute 1 Image
    ax[0][-1].imshow(imgs[1])
    ax[0][-1].axis('off')
    ax[0][-1].set_title('Attr. 1')

    # Set Attribute 2 Image
    ax[-1][0].imshow(imgs[2])
    ax[-1][0].axis('off')
    ax[-1][0].set_title('Attr. 2')

    # Generate J-Diagram Samples
    for i,t0 in enumerate(t0_vec):
        for j,t1 in enumerate(t1_vec):
            z_s = t0 * zba + t1 * zca + z0
            img_s = model.decode(tf.reshape(z_s, shape=(1, -1)), True)[0]
            
            ax_r = ax[j+1][i+1]
            ax_r.imshow(img_s)
            
    for a in ax:
        for b in a:
            b.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Save Figure to file
    print(f'Saving J-Diagram: {output_filename}')
    fig.savefig(output_filename, bbox_inches='tight')



def main():

    args = get_args()
    model, config = load_model(args.log_dir)
    imgs = load_images(args, config)

    plot_j_diagram(model, imgs, args.output_filename)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse, os, sys
import yaml

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from src.abstract_cvae import AbstractCVAE
from src.data_loader import load_data
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
    parser.add_argument('--output-path', '-o', type=str, default='umap_plot.png')
    parser.add_argument('--n-neighbors', '-n', type=int, default=15)
    parser.add_argument('--min-distance', '-d', type=float, default=0.1)
    parser.add_argument('--interpolate', '-i', action='store_true', help='Plot interpolation grid')
    parser.add_argument('--interpolation-output-filename', type=str, default='umap_interp.png')
    return parser.parse_args()


def plot_umap(data: dict, model: AbstractCVAE, output_path: str, n_neighbors: int, min_distance=float):

    z_train = []
    for batch in data['train']:
        z_train.append(model.call_detailed(batch)[1])
    z_train = tf.concat(z_train, axis=0)
    
    z_val = []
    for batch in data['val']:
        z_val.append(model.call_detailed(batch)[1])
    z_val = tf.concat(z_val, axis=0)

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_distance, verbose=True)
    umap_model.fit(z_train)

    train_embeddings = umap_model.transform(z_train)
    val_embeddings = umap_model.transform(z_val)

    fig, ax = plt.subplots(1, 1)

    fig.suptitle('UMAP Embeddings')
    ax.scatter(train_embeddings[:,0], train_embeddings[:,1], label='training', s=5)
    ax.scatter(val_embeddings[:,0], val_embeddings[:,1], label='validation', s=5)
    ax.legend()
    ax.grid()

    fig.savefig(output_path)

    return umap_model, train_embeddings, val_embeddings


def plot_interpolation(model: AbstractCVAE, umap_model: UMAP, train_embeddings: np.ndarray, val_embeddings: np.ndarray, output_filename:str):

    max_values = np.max(train_embeddings, axis=0)
    min_values = np.min(train_embeddings, axis=0)

    print(max_values, min_values)

    x_samples = np.linspace(min_values[0], max_values[0], 10)
    y_samples = np.linspace(min_values[1], max_values[1], 10)

    samples = []
    for x in x_samples:
        for y in y_samples:
            samples.append(np.array([x,y], dtype=np.float32))
    samples = np.array(samples, dtype=np.float32)

    z = umap_model.inverse_transform(samples)

    reconstructions = model.decode(z, True)

    fig, ax = plt.subplots(10,10)

    for i in range(10):
        for j in range(10):
            idx = i*10 + j
            ax[i][j].imshow(reconstructions[idx])
            ax[i][j].axis('off')
    
    fig.savefig(output_filename, bbox_inches='tight')


def main():

    args = get_args()
    model, config = load_model_from_directory(args.log_dir)
    data = load_data(config)
    umap_model, train_embeddings, val_embeddings = plot_umap(data, model, args.output_path, args.n_neighbors, args.min_distance)

    if args.interpolate:
        plot_interpolation(model, umap_model, train_embeddings, val_embeddings, args.interpolation_output_filename)


if __name__ == '__main__':
    main()

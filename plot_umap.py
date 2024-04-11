#!/usr/bin/env python3

import argparse, os, sys
import yaml
import tqdm

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
    parser.add_argument('--metric', '-m', type=str, default='euclidean', help='Distance metric (default=euclidean)')
    parser.add_argument('--standardize', '-s', action='store_true', help='Standardize latent space')
    parser.add_argument('--interpolate', '-i', action='store_true', help='Plot interpolation grid')
    parser.add_argument('--interpolation-output-filename', '-t', type=str, default='umap_interp.png')
    return parser.parse_args()


def plot_umap(data: dict, model: AbstractCVAE, output_path: str, n_neighbors: int, min_distance=float, dist_metric_str:str='euclidean', standardized_flag=False):

    z_train = []
    for batch in tqdm.tqdm(data['train'], desc='Get train latent space'):
        z_train.append(model.call_detailed(batch)[1])
    z_train = tf.concat(z_train, axis=0)
    
    z_val = []
    for batch in tqdm.tqdm(data['val'], desc='Get val latent space'):
        z_val.append(model.call_detailed(batch)[1])
    z_val = tf.concat(z_val, axis=0)

    stats = None
    if standardized_flag:
        print('Standardize latent space')
        z_mean = np.mean(z_train, axis=0)
        z_std = np.std(z_train, axis=0)

        z_train = (z_train - z_mean) / z_std
        z_val = (z_val - z_mean) / z_std
        stats = {'mean': z_mean, 'std': z_std}

    print('Fit UMAP model')
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_distance, metric=dist_metric_str, verbose=True)
    umap_model.fit(z_train)

    print('Get embeddings')
    train_embeddings = umap_model.transform(z_train)
    val_embeddings = umap_model.transform(z_val)

    print('Plot UMAP embeddings')
    fig, ax = plt.subplots(1, 1)

    fig.suptitle(f'UMAP Embeddings: Metric: {dist_metric_str}, Standardized: {standardized_flag}\nLatent Dim: {z_val.shape[1]}, N-Neighbors: {n_neighbors}, Min Dist: {min_distance}')
    ax.scatter(train_embeddings[:,0], train_embeddings[:,1], label='training', s=5)
    ax.scatter(val_embeddings[:,0], val_embeddings[:,1], label='validation', s=5)
    ax.legend()
    ax.grid()

    fig.savefig(output_path)

    return umap_model, train_embeddings, val_embeddings, stats


def plot_interpolation(model: AbstractCVAE, umap_model: UMAP, train_embeddings: np.ndarray, val_embeddings: np.ndarray, embedding_stats: dict, output_filename:str):

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

    #NOTE: UMAP appears to be broken for inverse transforms still - wmb
    z = umap_model.inverse_transform(samples)

    if embedding_stats:
        z = z * embedding_stats['std'] + embedding_stats['mean']

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
    umap_model, train_embeddings, val_embeddings, embedding_stats = plot_umap(data, model, args.output_path, args.n_neighbors, args.min_distance, args.metric, args.standardize)

    if args.interpolate:
        plot_interpolation(model, umap_model, train_embeddings, val_embeddings, embedding_stats, args.interpolation_output_filename)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import os
import tqdm
import json

from src.data_loader import load_data
from src.load_model import load_model_from_directory

import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from collections import defaultdict

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


def get_similarity_directory(output_dir: str):
    assert(os.path.isdir(output_dir))
    return os.path.join(output_dir, 'similarity')


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', '-m', type=str, required=True, help='Path to model directory')
    parser.add_argument('--dataset-dir', '-d', type=str, required=True, help='Path to dataset JSON')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Override output path')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing similarity output')

    args = parser.parse_args()

    assert(os.path.isdir(args.model_dir))
    assert(os.path.isdir(args.dataset_dir))
    if args.output_dir:
        assert(not os.path.isfile(args.output_dir))

    # Assign sim_dir based on if output_dir is specified
    if args.output_dir:
        sim_dir = get_similarity_directory(args.output_dir)
    else:
        sim_dir = get_similarity_directory(args.model_dir)

    # Check and create sim_dir
    if not args.force:
        assert(not os.path.exists(sim_dir))
        os.makedirs(sim_dir)
    else:
        # Clear out the directory if it exists for force overwrite
        if os.path.exists(sim_dir):
            assert(not os.path.isfile(sim_dir))
            import shutil
            shutil.rmtree(sim_dir)
        os.makedirs(sim_dir, exist_ok=True)

    return args.model_dir, args.dataset_dir, sim_dir


def similarity_analysis(model, config, data, sim_dir):

    '''
    #filepaths = [entry['full_filepath'] for entry in data['raite_db'].train_dict['images']]
    batch_size = config['training']['batch_size']
    
    import cv2

    fig, (ax0, ax1) = plt.subplots(1, 2)

    for batch_idx, (batch, batch_filenames) in enumerate(zip(data['train'], data['train_labels'])):
        for idx, (img_tensor, t_filename) in enumerate(zip(batch, batch_filenames)):

            file_idx = batch_idx * batch_size + idx
            #filename = filepaths[file_idx]
            filename = t_filename.numpy().decode('utf-8')
            print(filename)
            file_img = cv2.imread(filename)
            file_img = cv2.cvtColor(file_img, cv2.COLOR_BGR2RGB)

            ax0.clear()
            ax0.imshow(file_img)
            ax1.clear()
            ax1.imshow(img_tensor.numpy())
            fig.canvas.draw()
            plt.pause(0.5)

    '''

    # Filenames stored as tensors, so cast to str for processing
    cast_t_string = lambda ts: ts.numpy().decode('utf-8')

    z_samples = list()
    filepaths = list()

    for batch_img, batch_filepath in tqdm.tqdm(zip(data['train'], data['train_labels']), desc='Extracting samples'):
        _, z, _, _ = model.call_detailed(batch_img)
        z_samples.extend(z)
        filepaths.extend(batch_filepath)
    z_samples = np.array(z_samples)

    # Standardize the latent space
    print('Standardize latent space')
    latent_means = np.mean(z_samples, axis=0)
    latent_stds = np.std(z_samples, axis=0)
    z_scores = (z_samples - latent_means) / latent_stds

    euclidean_distance = pairwise_distances(z_scores, metric='euclidean')
    flat_euclidean_distance = euclidean_distance[np.triu_indices_from(euclidean_distance, k=1)]

    cosine_distance = pairwise_distances(z_scores, metric='cosine')
    flat_cosine_distance = cosine_distance[np.triu_indices_from(cosine_distance, k=1)]

    f1_f2_dist_lookup = defaultdict(lambda: defaultdict(dict))

    for idx_1, f1 in tqdm.tqdm(enumerate(filepaths), desc='Pairwise collection'):
        for idx_2, f2 in enumerate(filepaths):
            if idx_1 <= idx_2:
                continue
            f1_str = cast_t_string(f1)
            f2_str = cast_t_string(f2)
            f1_f2_dist_lookup[f1_str][f2_str]['euclidean'] = float(euclidean_distance[idx_1, idx_2])
            f1_f2_dist_lookup[f1_str][f2_str]['cosine'] = float(cosine_distance[idx_1, idx_2])

    distance_output_filepath = os.path.join(sim_dir, 'distances.json')
    print(f'Saving distances to file: {distance_output_filepath}')
    with open(distance_output_filepath, 'w', newline='') as ofile:
        json.dump(f1_f2_dist_lookup, ofile)

    print('Generating plots')
    fig, ((ax_euc, ax_full),(ax_cos, ax_each)) = plt.subplots(2, 2)

    fig.suptitle('Distance Metrics and Latent Space Histograms')

    ax_euc.hist(flat_euclidean_distance, bins='auto', density=True)
    ax_euc.set_title('Euclidean Distance Histogram')
    ax_euc.set_xlabel('Euclidean Distance')
    ax_euc.set_ylabel('Density')
    ax_euc.grid()

    ax_cos.hist(flat_cosine_distance, bins='auto', density=True)
    ax_cos.set_title('Cosine Distance Histogram')
    ax_cos.set_xlabel('Cosine Distance')
    ax_cos.set_ylabel('Density')
    ax_cos.grid()

    ax_full.hist(z_samples.flatten(), bins='auto', density=True)
    ax_full.set_title('Full Z Histogram')
    ax_full.set_xlabel('Latent Space Value')
    ax_full.set_ylabel('Density')
    ax_full.grid()

    for idx in range(z_samples.shape[1]):
        ax_each.hist(z_samples[:,idx], bins='auto', label=f'{idx} Vec', density=True, alpha=0.35)
    ax_each.set_title(f'Individual Z-vec Histogram: {z_samples.shape[1]}')
    ax_each.set_xlabel('Latent Space Value')
    ax_each.set_ylabel('Density')
    ax_each.grid()
    #ax_each.legend(loc='upper right', bbox_to_anchor=(1, 1), ncols=2)
    
    plt.tight_layout()

    figure_output_path = os.path.join(sim_dir, "similarity_figure.png")
    print(f'Output Figure: {figure_output_path}')
    fig.savefig(figure_output_path)

    plt.close()

    for idx in tqdm.tqdm(range(z_samples.shape[1]), desc='Latent Plots'):

        z_vec_mean = np.mean(z_samples[:,idx])
        z_vec_std = np.std(z_samples[:,idx])
        z_score = (z_samples[:,idx] - z_vec_mean) / z_vec_std
        kurtosis = np.mean(np.power(z_score, 4))

        z_fig, ax = plt.subplots(1, 1)
        z_fig.suptitle(f'Latent Element #:{idx} Histogram \nN= {len(z_score)}, Mean= {z_vec_mean:0.3f}, Std.Dev= {z_vec_std:0.3f}, Kurtosis= {kurtosis:0.3f}')
        ax.hist(z_samples[:,idx], bins='auto', density=True)
        ax.set_xlabel('Latent Space Value')
        ax.set_ylabel('Density')
        ax.grid()
        z_fig.savefig(os.path.join(sim_dir, f"latent_hist_{idx:03d}.png"))
        plt.close()

    #plt.show()



def main():

    model_dir, dataset_dir, output_dir = get_args()

    model, config = load_model_from_directory(model_dir)

    config['data']['dataset_path'] = dataset_dir
    data = load_data(config)

    similarity_analysis(model, config, data, output_dir)


if __name__ == '__main__':
    main()

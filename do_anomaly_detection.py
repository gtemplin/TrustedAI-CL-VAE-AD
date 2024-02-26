#!/usr/bin/env python3

import argparse
import os
import sys
import tqdm
import csv
from copy import deepcopy

import cv2
from PIL import Image

from src.abstract_cvae import AbstractCVAE
from src.data_loader import load_data
from src.load_model import load_model_from_directory

import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument('--model-dir', '-m', required=True, type=str, help='Model directory')
    parser.add_argument('--dataset-path', '-d', required=True, type=str, help='Dataset directory')
    parser.add_argument('--output-path', '-o', required=True, type=str, help='Output directory')
    parser.add_argument('--anomaly-threshold', '-t', type=float, default=3.0, help='Z-score thresh (default=3.0)')

    args = parser.parse_args()

    assert(os.path.exists(args.model_dir))
    assert(os.path.isdir(args.model_dir))
    assert(os.path.exists(args.dataset_path))
    assert(os.path.isdir(args.dataset_path))

    if os.path.exists(args.output_path):
        assert(os.path.isdir(args.output_path))
    
    os.makedirs(args.output_path, exist_ok=True)
    return args

def get_data_scale(model: AbstractCVAE, config: dict, data: dict):

    err_vec = list()
    for batch in tqdm.tqdm(data['train'], desc='Getting Training Outputs'):
        x_rec = model.call(tf.convert_to_tensor(batch), False)
        err = tf.reduce_sum(tf.pow(batch - x_rec, 2), axis=3)
        err_vec.append(err)
    err_vec = tf.concat(err_vec, axis=0)

    err_reduced = tf.reduce_sum(tf.reduce_sum(err_vec, axis=2), axis=1)
    meu = tf.reduce_mean(err_reduced)
    sigma = tf.math.reduce_std(err_reduced)
    z_scores = (err_reduced - meu) / sigma
    x_min = tf.reduce_min(err_vec)
    x_max = tf.reduce_max(err_vec)

    return {
        'meu': meu,
        'sigma': sigma,
        'min': x_min,
        'max': x_max,
        'z_scores': z_scores,
    }


def evaluate_anomalies(model: AbstractCVAE, config: dict, data:dict, data_scale:dict, anomaly_threshold: float):

    x_rec_vec, err_vec, z_scores, norm_err = list(), list(), list(), list()
    for batch in tqdm.tqdm(data['train'], desc='Evaluate Anomalies'):
        x_rec, _, _, _ = model.call_detailed(batch)

        err = tf.reduce_sum(tf.pow(batch - x_rec, 2), axis=3)
        
        err_reduced = tf.reduce_sum(tf.reduce_sum(err, axis=2), axis=1)
        z_score = (err_reduced - data_scale['meu']) / data_scale['sigma']
        norm_batch = (err - data_scale['min']) / (data_scale['max'] - data_scale['min'])
        
        x_rec_vec.append(x_rec)
        err_vec.append(err)
        z_scores.append(z_score)
        norm_err.append(norm_batch)

    x_rec = tf.concat(x_rec_vec, axis=0)
    err_vec = tf.concat(err_vec, axis=0)
    z_scores = tf.concat(z_scores, axis=0)
    norm_err = tf.concat(norm_err, axis=0)

    z_scores_np = z_scores.numpy()
    anomalies = z_scores_np > anomaly_threshold

    print(anomalies)
    print(np.sum(anomalies))
    print(np.sum(anomalies) / len(anomalies))

    return {
        'rec': x_rec.numpy(),
        'errs': err_vec.numpy(),
        'z_scores': z_scores.numpy(),
        'norm_errs': norm_err.numpy(),
        'anomalies': anomalies,
    }



def output_anomalies(evaluation_data, anomaly_results: dict, data_scale: dict, output_path: str, anomaly_threshold: float):

    assert(os.path.exists(output_path))
    assert(os.path.isdir(output_path))

    err_path = os.path.join(output_path, 'err')
    heatmap_path = os.path.join(output_path, 'heatmap')
    overlay_path = os.path.join(output_path, 'overlay')
    rec_path = os.path.join(output_path, 'rec')
    orig_path = os.path.join(output_path, 'orig')
    anomaly_hist_path = os.path.join(output_path, 'anomaly_fig.png')
    anomaly_csv_path = os.path.join(output_path, 'anomaly_list.csv')


    os.makedirs(err_path, exist_ok=True)
    os.makedirs(heatmap_path, exist_ok=True)
    os.makedirs(overlay_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(orig_path, exist_ok=True)


    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Error Z-Score Histogram (Per Frame)')
    ax.hist(data_scale['z_scores'].numpy(), bins='auto', label='Still Data', alpha=0.45, density=True)
    ax.hist(anomaly_results['z_scores'], bins='auto', label='Evaluation Data', alpha=0.45, density=True)
    ax.axvline(anomaly_threshold, color='red', alpha=0.85)
    ax.set_xlim(-3.0, 70.0)
    ax.set_xlabel('Z-Score (Normal Assumption)')
    ax.set_ylabel('Density (Per Frame)')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    plt.tight_layout()

    fig.savefig(anomaly_hist_path)
    exit()

    orig_filename_list = []
    x = evaluation_data['train'].unbatch()
    for i,(x_entry, rec, norm_err) in tqdm.tqdm(enumerate(zip(x, anomaly_results['rec'], anomaly_results['norm_errs'])), desc='Output Anomaly Files'):

        x_i = tf.convert_to_tensor(x_entry)

        err_numpy = np.round(255. * norm_err).astype(np.uint8)
        heatmap = cv2.applyColorMap(err_numpy, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, np.round(255. * rec).astype(np.uint8), 0.5, 0.0)

        err_img = Image.fromarray(err_numpy, mode='L')
        heatmap_img = Image.fromarray(heatmap, mode='RGB')
        overlay_img = Image.fromarray(overlay, mode='RGB')
        rec_img = Image.fromarray(rec, mode='RGB')
        orig_img = Image.fromarray(np.round(255. * x_i.numpy()).astype(np.uint8), mode='RGB')

        basename = f'{i:06d}.png'
        orig_filename = os.path.join(orig_path, basename)
        err_filename = os.path.join(err_path, basename)
        heatmap_filename = os.path.join(heatmap_path, basename)
        overlay_filename = os.path.join(overlay_path, basename)
        rec_filename = os.path.join(rec_path, basename)

        err_img.save(err_filename)
        heatmap_img.save(heatmap_filename)
        overlay_img.save(overlay_filename)
        rec_img.save(rec_filename)
        orig_img.save(orig_filename)

        orig_filename_list.append(orig_filename)

    output_tuples = sorted(zip(orig_filename_list, anomaly_results['z_scores']), key=lambda x: x[1], reverse=True)
        
    with open(anomaly_csv_path, 'w', newline='') as ofile:
        writer = csv.writer(ofile)
        writer.writerow(['orig_filepath', 'z_score'])
        for row in tqdm.tqdm(output_tuples, desc='Write out anomaly csv'):
            writer.writerow(row)

    print(f'Anomalies written out to: {output_path}')




def main():

    args = get_args()

    model_dir = args.model_dir
    dataset_path = args.dataset_path
    output_path = args.output_path
    anomaly_threshold = args.anomaly_threshold

    model, config = load_model_from_directory(model_dir)

    train_data = load_data(config)
    data_scale = get_data_scale(model, config, train_data)

    config['data']['dataset_path'] = dataset_path
    
    evaluation_data = load_data(config)

    anomaly_results = evaluate_anomalies(model, config, evaluation_data, data_scale, anomaly_threshold)
    output_anomalies(evaluation_data, anomaly_results, data_scale, output_path, anomaly_threshold)



if __name__ == '__main__':
    main()
#!/usr/bin/env python3

#!/usr/bin/env python

import argparse, os
import tensorflow as tf
import yaml
import tqdm
import json
import numpy as np

import cv2

from multiprocessing import Pool
from itertools import repeat

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from src.fuzzy_vae import FuzzyVAE
from src.data_loader import load_data


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
    parser.add_argument('log_dir', type=str, help='Model Directory')
    
    return parser.parse_args()

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


def process_train_val_reconstructions(log_dir: str, model: FuzzyVAE, config: dict, data: dict):

    assert(os.path.exists(log_dir))
    assert(os.path.isdir(log_dir))

    # Create directories
    train_original_dir = os.path.join(log_dir, 'imgs/originals/train')
    val_original_dir = os.path.join(log_dir, 'imgs/originals/val')
    train_reconstructions_dir = os.path.join(log_dir, 'imgs/reconstructions/train')
    val_reconstructions_dir = os.path.join(log_dir, 'imgs/reconstructions/val')
    train_heatmap_dir = os.path.join(log_dir, 'imgs/heatmap/train')
    val_heatmap_dir = os.path.join(log_dir, 'imgs/heatmap/val')
    train_error_dir = os.path.join(log_dir, 'imgs/errors/train')
    val_error_dir = os.path.join(log_dir, 'imgs/errors/val')

    batchsize = int(config['training']['batch_size'])

    #def _mp_get_recon(idx, x, x_hat, orig_path, recon_path):
    #    img_num = batch_id*batchsize + idx

    def _draw_heatmap(orig_img, rec_error_norm, min_error, max_error):
        heatmap = cv2.applyColorMap(rec_error_norm, cv2.COLORMAP_JET)
        superimpose = cv2.addWeighted(heatmap, 0.5, orig_img, 0.5, 0.0)
        return superimpose
    
    def _draw_heatmaps(orig_img_dir, rec_error_img_dict, _heat_dir, _error_dir):
        min_error = tf.reduce_min([r for r in rec_error_img_dict.values()])
        max_error = tf.reduce_max([r for r in rec_error_img_dict.values()])
        for img_num, rec_error in tqdm.tqdm(rec_error_img_dict.items(), desc='Drawing Heatmaps'):
            rec_error_norm = tf.cast(tf.round(255.0 * (rec_error - min_error) / (max_error - min_error)), dtype=tf.uint8).numpy()
            orig_img = np.array(tf.keras.utils.load_img(os.path.join(orig_img_dir, f'{img_num}.png')))
            #orig_img = tf.cast(tf.round(255.0 * orig_img_dict[img_num]), dtype=tf.uint8).numpy()
            heatmap = _draw_heatmap(orig_img, rec_error_norm, min_error, max_error)
            heatmap_path = os.path.join(_heat_dir, f'{img_num}.png')
            tf.keras.utils.save_img(heatmap_path, heatmap)
            error_path = os.path.join(_error_dir, f'{img_num}.png')
            tf.keras.utils.save_img(error_path, tf.reshape(rec_error_norm, shape=tuple(rec_error_norm.shape) + (1,)))


    def _draw_reconstructions(_model: FuzzyVAE, _data: tf.data.Dataset, _orig_dir:str, _rec_dir:str, _heat_dir:str, _error_dir:str, _batchsize:int, tqdm_msg:str):
        os.makedirs(_orig_dir)
        os.makedirs(_rec_dir)
        os.makedirs(_heat_dir)
        os.makedirs(_error_dir)

        rec_error_img_dict = {}
        rec_err_dict = {}
        #orig_img_dict = {}

        for batch_id, batch in tqdm.tqdm(enumerate(_data), desc=tqdm_msg):
            x_hat = _model.call(tf.convert_to_tensor(batch), False)

            for idx, (x, x_hat) in enumerate(zip(batch, x_hat)):

                img_num = batch_id * _batchsize + idx
                orig_path = os.path.join(_orig_dir, f'{img_num}.png')
                rec_path = os.path.join(_rec_dir, f'{img_num}.png')

                #orig_img_dict[img_num] = x
                rec_error_img_dict[img_num] = tf.math.reduce_sum(tf.math.pow(tf.math.subtract(x, x_hat), 2), axis=2)
                rec_err_dict[img_num] = float(tf.math.sqrt(tf.math.reduce_sum(rec_error_img_dict[img_num])).numpy())

                tf.keras.utils.save_img(orig_path, x)
                tf.keras.utils.save_img(rec_path, x_hat)

        _draw_heatmaps(_orig_dir, rec_error_img_dict, _heat_dir, _error_dir)

        return rec_error_img_dict, rec_err_dict

    _ , train_reconstruction_error = _draw_reconstructions(model, data['train'], train_original_dir, train_reconstructions_dir, train_heatmap_dir, train_error_dir, batchsize, 'Drawing Training Set')
    with open(os.path.join(log_dir, 'train_reconstruction_error.json'), 'w') as ofile:
        json.dump(train_reconstruction_error, ofile)

    _ , val_reconstruction_error = _draw_reconstructions(model, data['val'], val_original_dir, val_reconstructions_dir, val_heatmap_dir, val_error_dir, batchsize, 'Drawing Validation Set')
    with open(os.path.join(log_dir, 'val_reconstruction_error.json'), 'w') as ofile:
        json.dump(val_reconstruction_error, ofile)
    

    fig, ax = plt.subplots(1, 1)

    fig.suptitle('Reconstruction Error Histogram')
    ax.hist([r for r in train_reconstruction_error.values()], label='train', bins='auto')
    ax.hist([r for r in val_reconstruction_error.values()], label='val', bins='auto')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.grid()
    ax.legend()

    fig.savefig(os.path.join(log_dir, 'reconstruction_hist.png'))


def main():

    args = get_args()
    model, config = load_model(args.log_dir)

    data = load_data(config)

    process_train_val_reconstructions(args.log_dir, model, config, data)

if __name__ == '__main__':
    main()

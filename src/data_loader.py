#!/usr/bin/env python3

import os
import tensorflow as tf
import tensorflow_datasets as tfds

from src.raite_loader import RaiteDataset


def _normalize_img(element):
    return tf.cast(element['image'], tf.float32) / 255.

def _resize_img(element, img_size):
    return tf.image.resize(element, size=img_size, antialias=True)

def load_data(config: dict):

    data_config = config['data']
    dataset_path = data_config.get('dataset_path')
    dataset_name = data_config.get('dataset')
    train_split = data_config['train_split']
    val_split = data_config['val_split']
    img_size = data_config['image_size']
    batch_size = config['training']['batch_size']

    r_img_size = (img_size[0], img_size[1])

    if dataset_name == 'raite':
        assert(os.path.exists(dataset_path))
        assert(os.path.isdir(dataset_path))
        print(f'Loading RAITE Dataset from: {dataset_path}')

        train_path = os.path.join(dataset_path, train_split)
        print(f'Train Path: {train_path}')
        assert(os.path.exists(train_path))
        
        test_path = os.path.join(dataset_path, val_split)
        print(f'Validation Path: {test_path}')
        assert(os.path.exists(test_path))

        rdb = RaiteDataset(train_path, test_path)

        rdb.train_data = rdb.train_data.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        rdb.test_data = rdb.test_data.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
        rdb.train_data = rdb.train_data.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE).cache()
        rdb.test_data = rdb.test_data.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE).cache()

        return {'train': rdb.train_data, 'val': rdb.test_data, 'raite_db': rdb}

    if dataset_path is not None:
        print(f'Loading dataset from: {dataset_path}')
        assert(os.path.exists(dataset_path))
        assert(os.path.isdir(dataset_path))
        
        train_ds = tf.data.Dataset.load(os.path.join(dataset_path, 'train'))
        val_ds = tf.data.Dataset.load(os.path.join(dataset_path, 'validation'))
        
        train_ds = train_ds.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        train_ds, ds_info = tfds.load(dataset_name, split=train_split, shuffle_files=True, download=False, with_info=True)
        val_ds = tfds.load(dataset_name, split=val_split, shuffle_files=True, download=False, with_info=False)
        
        
        train_ds = train_ds.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    print(f'Length of Training data: {len(train_ds)}')

    return {
        'train': train_ds,
        'val': val_ds,
        #'info': ds_info,
    }


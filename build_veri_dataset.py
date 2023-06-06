#!/usr/env/bin python3

import argparse, os
import tqdm

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
    parser.add_argument("train_path", type=str)
    parser.add_argument("val_path", type=str)
    parser.add_argument("--output-path", "-o", type=str, default="VeRi_dataset")
    args = parser.parse_args()

    return args


def load_data(data_path: str):

    assert(os.path.exists(data_path))
    assert(os.path.isdir(data_path))

    ds = tf.keras.utils.image_dataset_from_directory(data_path, labels=None, color_mode='rgb', batch_size=32, image_size=(224,224))
    ds = ds.unbatch()

    ds = ds.map(lambda x: {'image': x})

    return ds


def build_data(train_path: str, val_path: str):

    train_ds = load_data(train_path)
    val_ds = load_data(val_path)
    return train_ds, val_ds

def save_data(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, output_path: str):

    os.makedirs(output_path)
    train_ds.save(os.path.join(output_path, 'train'))
    val_ds.save(os.path.join(output_path, 'validation'))


def main():
    args = get_args()
    train_ds, val_ds = build_data(args.train_path, args.val_path)
    save_data(train_ds, val_ds, args.output_path)


if __name__ == '__main__':
    main()


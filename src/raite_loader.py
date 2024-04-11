#!/usr/bin/env python3

import argparse
import os
import sys
import json
from collections import defaultdict
import time
import tqdm

import cv2

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


class RaiteDataset(object):

    def __init__(self, train_json_path, test_json_path, batchsize=32):

        super(RaiteDataset, self).__init__()

        self.train_dict = self._load_json_data(train_json_path)
        self.test_dict = self._load_json_data(test_json_path)

        #output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.int8), tf.RaggedTensorSpec(shape=[None, 4], dtype=tf.float32))
        #output_signature = tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
        output_signature = (tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name='image'), tf.TensorSpec(shape=(), dtype=tf.string, name='filepath'))

        self.train_data = tf.data.Dataset.from_generator(self._build_tf_data, args=('train',), output_signature=output_signature)
        self.test_data = tf.data.Dataset.from_generator(self._build_tf_data, args=('test', ), output_signature=output_signature)

        #def _map_func(img, bbox):
        #    return {'image': img, 'bbox_list': bbox}
        def _map_func(entry0, entry1):
            return {
                'image': entry0,
                'filepath': entry1,
                }

        self.train_data = self.train_data.map(_map_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batchsize)
        self.test_data = self.test_data.map(_map_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batchsize)

        #dataset = tf.data.Dataset.range(2).interleave(lambda _ : dataset, num_parallel_calls=tf.data.AUTOTUNE).cache()
        self.train_data = tf.data.Dataset.range(2).interleave(lambda _: self.train_data.prefetch(tf.data.AUTOTUNE), num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = tf.data.Dataset.range(2).interleave(lambda _: self.test_data.prefetch(tf.data.AUTOTUNE), num_parallel_calls=tf.data.AUTOTUNE)

        #self.train_data = self.train_data.batch(32)
        #for row in self.train_data:
        #    print(row['image'].shape, row['bbox_list'].shape, row['bbox_list'][0])


    def _load_json_data(self, json_data_path):
        assert(os.path.exists(json_data_path))
        assert(os.path.isfile(json_data_path))

        data = None

        try:
            with open(json_data_path, 'r') as ifile:
                data = json.load(ifile)
        except IOError as e:
            raise e
        except Exception as e:
            raise e
        
        assert(data is not None)

        json_dirname = os.path.abspath(os.path.dirname(json_data_path))
        img_dirname = os.path.join(json_dirname, "frames")
        assert(os.path.exists(img_dirname))
        assert(os.path.isdir(img_dirname))

        for row in data['images']:
            image_filepath = os.path.join(img_dirname, row['file_name'])
            assert(os.path.exists(image_filepath))
            assert(os.path.isfile(image_filepath))
            row['full_filepath'] = image_filepath

        return data
    

    def _build_tf_data(self, dataset_selection: str, dataset_task: str='scene'):

        # These get encoded as binary
        if type(dataset_selection) is bytes:
            dataset_selection = dataset_selection.decode('utf-8')
        if type(dataset_task) is bytes:
            dataset_task = dataset_task.decode('utf-8')

        if dataset_selection not in ['train', 'test']:
            raise RuntimeError(f'Error, unrecognized argument: {dataset_selection} (["test", "train"])')
        
        if dataset_task not in ['scene', 'objects']:
            raise RuntimeError(f'Error, unrecognized argument: {dataset_task} (["scene", "objects"])')

        data_dict = None
        if dataset_selection == 'train':
            data_dict = self.train_dict
        elif dataset_selection == 'test':
            data_dict = self.test_dict

        assert(data_dict is not None)

        image_order_to_image_id_map = {i: row['id'] for i,row in enumerate(data_dict['images'])}
        #print(len(image_order_to_image_id_map))
        #print(len(data_dict['images']))

        image_id_annotation_map = defaultdict(list)
        for idx, annotation in enumerate(data_dict['annotations']):
            image_id = annotation['image_id']
            image_id_annotation_map[image_id].append(idx)

        #print(len(data_dict['annotations']))
        #print(len(image_id_annotation_map))
        
        for idx, image_meta in enumerate(data_dict['images']):
            img_filepath = image_meta['full_filepath']

            try:
                img = cv2.imread(img_filepath)
            except Exception:
                continue

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #TODO: bbox retrieval is incorrect, not robust -wmb
            #image_id = image_order_to_image_id_map[idx]
            #annotation_ids = image_id_annotation_map[image_id]
            #annotations = [data_dict['annotations'][ann_id] for ann_id in annotation_ids]
            #bbox = tf.constant([row['bbox'] for row in annotations], dtype=tf.float32)

            ## Placeholder for empties
            #if len(bbox) == 0:
            #    bbox = tf.constant([[-1, -1, -1, -1]], dtype=tf.float32)

            #yield img, tf.RaggedTensor.from_tensor(bbox)
            yield img, img_filepath

    def _split_data_labels(self, db):
        data = db.map(lambda x: x['image'], num_parallel_calls=tf.data.AUTOTUNE)
        labels = db.map(lambda x: x['filepath'], num_parallel_calls=tf.data.AUTOTUNE)
        return data, labels
    def split_train_data_labels(self):
        print('Split Train Labels')
        return self._split_data_labels(self.train_data)
    def split_test_data_labels(self):
        print('Split Test Labels')
        return self._split_data_labels(self.test_data)



def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_json_path', type=str, help='Train File Location')
    parser.add_argument('test_json_path', type=str, help='Test File Location')
    parser.add_argument('--benchmark-epochs', '-e', type=int, default=2, help='Number of epochs to benchmark (default=2)')

    return parser.parse_args()


def main():

    import matplotlib.pyplot as plt

    args = get_args()    
    db = RaiteDataset(args.train_json_path, args.test_json_path)

    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in tqdm.tqdm(dataset, desc=f'Epoch: {epoch_num}'):
                pass
        print(f'Execution Time: {time.perf_counter() - start_time}')

    print('Executing Benchmark')

    def _normalize_img(element):
        return tf.cast(element['image'], tf.float32) / 255.

    def _resize_img(element, img_size):
        return tf.image.resize(element, size=img_size, antialias=True)
    
    r_img_size = (224, 224)

    db.train_data = db.train_data.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    db.test_data = db.test_data.map(_normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    db.train_data = db.train_data.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE).cache()
    db.test_data = db.test_data.map(lambda x: _resize_img(x, r_img_size), num_parallel_calls=tf.data.AUTOTUNE).cache()

    for x in db.train_data.take(1):
        img = x[0]
        print(tf.math.reduce_max(img))
        print(tf.math.reduce_min(img))
        print(img)
        plt.imshow(img)
        plt.show()
        exit()


    print('Training Set')
    benchmark(db.train_data, args.benchmark_epochs)
    print('Test Set')
    benchmark(db.test_data, args.benchmark_epochs)

    print('Benchmark complete')



if __name__ == '__main__':
    main()

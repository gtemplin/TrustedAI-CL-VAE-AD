#!/usr/bin/env python3

import argparse, os
import tqdm

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from itertools import repeat

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
    parser.add_argument('virat_directory', type=str, help='The directory for Ground data')
    parser.add_argument('--output-path', '-o', type=str, default='virat_dataset', help='Output directory path for TF dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Batchsize for processing frames and annotations')
    args = parser.parse_args()

    return args

def load_meta_data(virat_directory: str):

    assert(os.path.exists(virat_directory))
    assert(os.path.isdir(virat_directory))

    virat_directory = os.path.abspath(virat_directory)

    annotations_dir = os.path.join(virat_directory, 'annotations')
    videos_dir = os.path.join(virat_directory, 'videos_original')

    assert(os.path.exists(annotations_dir))
    assert(os.path.isdir(annotations_dir))

    assert(os.path.exists(videos_dir))
    assert(os.path.isdir(videos_dir))

    meta = dict()

    for dirpath, dirnames, filenames in os.walk(videos_dir):
        
        for filename in filenames:
            basename, ext = os.path.splitext(filename)

            if ext.lower() == '.mp4':
                events_path = os.path.join(annotations_dir, f'{basename}.viratdata.events.txt')
                mapping_path = os.path.join(annotations_dir, f'{basename}.viratdata.mapping.txt')
                objects_path = os.path.join(annotations_dir, f'{basename}.viratdata.objects.txt')

                if not os.path.isfile(events_path):
                    print(f'No Events File: {basename}')
                    events_path = None
                if not os.path.isfile(mapping_path):
                    print(f'No Mapping File: {basename}')
                    mapping_path = None
                if not os.path.isfile(objects_path):
                    print(f'No Object File: {basename}')
                    objects_path = None

                meta[basename] = {
                    'video_path': os.path.join(dirpath, filename),
                    'events_path': events_path,
                    'mapping_path': mapping_path,
                    'objects_path': objects_path,
                }

    print(len(meta))

    for basename, obj in meta.items():
        print(f'Basename: {basename}')
        for key,val in obj.items():
            print(f'{key}: {val}')

    return meta


def get_event_annotations_from_file(event_path: str):

    if event_path is None:
        return None
    if not os.path.exists(event_path):
        return None
    if not os.path.isfile(event_path):
        return None
    
    event_list = list()
    with open(event_path, 'r') as ifile:
        for row in ifile.readlines():
            s = row.split()
            event_list.append({
                'event_id': int(s[0]),
                'event_type': int(s[1]),
                'duration': int(s[2]),
                'start_frame': int(s[3]),
                'end_frame': int(s[4]),
                'current_frame': int(s[5]),
                'bbox_lefttop_x': int(s[6]),
                'bbox_lefttop_y': int(s[7]),
                'bbox_width': int(s[8]),
                'bbox_height': int(s[9]),
            })

    return event_list


def get_mapping_annotations_from_file(mapping_path: str):

    if mapping_path is None:
        return None
    if not os.path.exists(mapping_path):
        return None
    if not os.path.isfile(mapping_path):
        return None
    
    mapping_list = list()
    with open(mapping_path, 'r') as ifile:
        for row in ifile.readlines():
            s = row.split()
            mapping_list.append({
                'event_id': int(s[0]),
                'event_type': int(s[1]),
                'duration': int(s[2]),
                'start_frame': int(s[3]),
                'end_frame': int(s[4]),
                'num_objects': int(s[5]),
                'obj_col_map': [ int(i) for i in s[6:] ],
            })

    return mapping_list


def get_object_annotations_from_file(objects_path: str):

    if objects_path is None:
        return None
    if not os.path.exists(objects_path):
        return None
    if not os.path.isfile(objects_path):
        return None
    
    objects_list = list()
    with open(objects_path, 'r') as ifile:
        for row in ifile.readlines():
            s = row.split()
            objects_list.append({
                'obj_id': int(s[0]),
                'duration': int(s[1]),
                'current_frame': int(s[2]),
                'bbox_lefttop_x': int(s[3]),
                'bbox_lefttop_y': int(s[4]),
                'bbox_width': int(s[5]),
                'bbox_height': int(s[6]),
                'obj_type': int(s[7]),
            })

    return objects_list


def parse_annotations(meta_data:dict):

    annotations = {}

    for basename, obj in tqdm.tqdm(meta_data.items(), desc='Grabbing annotations'):
        event_annotations = get_event_annotations_from_file(obj['events_path'])
        mapping_annotations = get_mapping_annotations_from_file(obj['mapping_path'])
        object_annotations = get_object_annotations_from_file(obj['objects_path'])

        annotations[basename] = {
            'events': event_annotations,
            'mapping': mapping_annotations,
            'objects': object_annotations,
        }

    return annotations


def parse_video_name_data(basename: str):
    
    basename_segments = basename.split('_')
    group_id = None
    scene_id = None
    sequence_id = None
    segment_id = None
    start_seconds = None
    end_seconds = None

    if len(basename_segments) >= 2:
        group_id = int(basename_segments[2][0:2])
        scene_id = int(basename_segments[2][2:4])
        sequence_id = int(basename_segments[2][4:6])

    # Baseline Scense are missing meta data
    if len(basename_segments) >= 6:
        segment_id = int(basename_segments[3])
        start_seconds = int(basename_segments[4])
        end_seconds = int(basename_segments[5])

    return {
        'basename': basename,
        'group_id': group_id,
        'scene_id': scene_id,
        'sequence_id': sequence_id,
        'segment_id': segment_id,
        'start_seconds': start_seconds,
        'end_seconds': end_seconds,
    }


def build_event_frame_map(annotations_entry: dict):
    event_frame_map = defaultdict(list)
    if annotations_entry.get('events') is not None:
        for idx, event_entry in enumerate(annotations_entry['events']):
            event_frame_map[event_entry['current_frame']].append(idx)
    return event_frame_map


def build_object_frame_map(annotations_entry: dict):
    obj_frame_map = defaultdict(list)
    if annotations_entry.get('objects') is not None:
        for idx, obj_entry in enumerate(annotations_entry['objects']):
            obj_frame_map[obj_entry['current_frame']].append(idx)
    return obj_frame_map


def serialize_frame_events(frame_id: int, event_frame_map: dict, annotations_entry: dict):
    event_labels = list()
    if event_frame_map.get(frame_id) is not None:
        for event_entry_idx in event_frame_map.get(frame_id):
            event_labels.append(annotations_entry['events'][event_entry_idx])
    
    event_features = {}
    if len(event_labels) > 0:
        for key in event_labels[0].keys():
            event_features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[
                event_row[key] for event_row in event_labels
            ]))
    return tf.train.Features(feature=event_features).SerializeToString()


def serialize_frame_objects(frame_id: int, obj_frame_map: dict, annotations_entry: dict):
    object_labels = list()
    if obj_frame_map.get(frame_id) is not None:
        for object_entry_idx in obj_frame_map.get(frame_id):
            object_labels.append(annotations_entry['objects'][object_entry_idx])

    obj_features = {}
    if len(object_labels) > 0:
        for key in object_labels[0].keys():
            obj_features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[
                obj_row[key] for obj_row in object_labels
            ]))
    return tf.train.Features(feature=obj_features).SerializeToString()


def _to_bytelist(feature):
    if feature is None:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
def _to_floatlist(feature):
    if feature is None:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[]))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))
def _to_int64list(feature):
    if feature is None:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))
    
def serialize_complete_frame(frame_id: int, video_name_data: dict, event_frame_map: dict, obj_frame_map: dict, annotations_entry: dict):
    
    # Serialize Event Features
    event_features = serialize_frame_events(frame_id, event_frame_map, annotations_entry)

    # Serialize Object Features
    obj_features = serialize_frame_objects(frame_id, obj_frame_map, annotations_entry)

    return tf.train.Example(features=tf.train.Features(feature={
        #"image": _to_bytelist(frame.tobytes()),
        "basename": _to_bytelist(video_name_data['basename'].encode("utf-8")),
        "group_id": _to_int64list(video_name_data['group_id']),
        "scene_id": _to_int64list(video_name_data['scene_id']),
        "sequence_id": _to_int64list(video_name_data['sequence_id']),
        "segment_id": _to_int64list(video_name_data['segment_id']),
        "start_seconds": _to_int64list(video_name_data['start_seconds']),
        "end_seconds": _to_int64list(video_name_data['end_seconds']),
        "events": _to_bytelist(event_features),
        "objects": _to_bytelist(feature=obj_features),
    })).SerializeToString()
    

def flatten_frames_and_annotations(ofile: tf.io.TFRecordWriter, basename: str, meta_data: dict, annotations: dict):

    meta_data_entry = meta_data[basename]
    annotations_entry = annotations[basename]

    video_path = meta_data_entry.get('video_path')

    if video_path is None:
        return None
    if not os.path.exists(video_path):
        return None
    if not os.path.isfile(video_path):
        return None
    
    # Get Group/Scene/Sequence/Segment/Time Meta from Filename
    video_name_data = parse_video_name_data(basename)

    # Get Event Frame/Index Map
    event_frame_map = build_event_frame_map(annotations_entry)

    # Get Ojbect Frame/Index Map
    obj_frame_map = build_object_frame_map(annotations_entry)

    # Load video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(None, desc=f'Loading Frames: {basename}', total=total_frames)

    records = list()

    frame_id = 0
    while cap.isOpened():

        # Get Frames
        ret, frame = cap.read()
        if ret == 0:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        record_bytes = serialize_complete_frame(frame_id, video_name_data, event_frame_map, obj_frame_map, annotations_entry)
        ofile.write(record_bytes)
        #records.append(deepcopy(record_bytes))

        progress_bar.update()
        frame_id += 1

    progress_bar.close()
    if cap.isOpened():
        cap.release()
    
    #return records
    

def restructure_tf_dataset(tf_dataset:tf.data.TFRecordDataset):

    print(tf_dataset)

    for row in tf_dataset:
        example = tf.train.Example()
        example.ParseFromString(row.numpy())
        print(example)

    return tf_dataset


def conjunction_of_spheres(meta_data: dict, annotations: dict, output_path: str):
    
    buffer_file = os.path.join(os.path.dirname(output_path), 'buffer.tfrecord')
    assert(not os.path.exists(buffer_file))

    try:
        with tf.io.TFRecordWriter(buffer_file) as ofile:
            for basename in tqdm.tqdm(meta_data.keys(), desc='Joining frames/anns'):
                flatten_frames_and_annotations(ofile, basename, meta_data, annotations)

        tf_dataset = tf.data.TFRecordDataset([buffer_file])
        #tf_dataset = restructure_tf_dataset(tf_dataset)
        tf_dataset.save(output_path)
    finally:
        os.remove(buffer_file)


    return tf_dataset


def create_tf_dataset(meta_data: dict, output_path: str):
    
    annotations = parse_annotations(meta_data)
    return conjunction_of_spheres(meta_data, annotations, output_path)



def main():

    args = get_args()
    meta_data = load_meta_data(args.virat_directory)
    tf_dataset = create_tf_dataset(meta_data, args.output_path)




if __name__ == '__main__':
    main()

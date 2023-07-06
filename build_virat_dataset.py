#!/usr/bin/env python3

import argparse, os
import tqdm

from collections import defaultdict

import cv2
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


def get_event_annotations(event_path: str):

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


def get_mapping_annotations(mapping_path: str):

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


def get_object_annotations(objects_path: str):

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
        event_annotations = get_event_annotations(obj['events_path'])
        mapping_annotations = get_mapping_annotations(obj['mapping_path'])
        object_annotations = get_object_annotations(obj['objects_path'])

        annotations[basename] = {
            'events': event_annotations,
            'mapping': mapping_annotations,
            'objects': object_annotations,
        }

    return annotations


def flatten_frames_and_annotations(basename: str, meta_data: dict, annotations: dict, batchsize: int):

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

    # Get Event Frame/Index Map
    event_frame_map = defaultdict(list)
    for idx, event_entry in enumerate(annotations_entry['events']):
        event_frame_map[event_entry['current_frame']].append(idx)

    # Get Ojbect Frame/Index Map
    obj_frame_map = defaultdict(list)
    for idx, obj_entry in enumerate(annotations_entry['objects']):
        obj_frame_map[obj_entry['current_frame']].append(idx)

    # Load video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return None
    
    frames_tf = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm.tqdm(None, desc=f'Loading Frames: {basename}', total=total_frames)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == 0:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frames_tf is None:
            frames_tf = tf.convert_to_tensor([frame], dtype=tf.uint8)
        else:
            frames_tf = tf.concat((frames_tf, tf.convert_to_tensor([frame], dtype=tf.uint8)), 0)

        progress_bar.update()

        frame_id += 1

    print(frames_tf.shape)

    progress_bar.close()
    if cap.isOpened():
        cap.release()
    

    exit()
    




def conjunction_of_spheres(meta_data: dict, annotations: dict, batchsize):

    for basename in tqdm.tqdm(meta_data.keys(), desc='Joining frames/anns'):
        r = flatten_frames_and_annotations(basename, meta_data, annotations, batchsize)


        


def create_tf_dataset(meta_data: dict, batchsize=32):
    
    annotations = parse_annotations(meta_data)
    return conjunction_of_spheres(meta_data, annotations, batchsize)


def save_dataset(tf_dataset: tf.data.Dataset, output_path: str):
    pass



def main():

    args = get_args()
    meta_data = load_meta_data(args.virat_directory)
    tf_dataset = create_tf_dataset(meta_data, args.batchsize)
    save_dataset(tf_dataset, args.output_path)




if __name__ == '__main__':
    main()

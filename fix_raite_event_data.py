#!/usr/bin/env python3

import argparse
import os
import sys
import csv
import tqdm
import datetime
import re

import cv2

from multiprocessing import Pool
from itertools import repeat
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from build_raite_json_from_directory import build_config_from_directory


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='Path to "raite_event" root dir')
    parser.add_argument('--output-dir', '-o', required=True, type=str, help='Path to output directory (it\'s large so no default)')
    parser.add_argument('--force', '-f', action='store_true', help='Force overwrite of output')
    args = parser.parse_args()

    assert(os.path.exists(args.root_dir))
    assert(os.path.isdir(args.root_dir))

    if os.path.exists(args.output_dir):
        if not args.force:
            print(f'Error, output path exists (call --force to overwrite): {args.output_dir}', file=sys.stderr)
            exit(1)
        else:
            assert(os.path.isdir(args.output_dir))
    else:
        os.makedirs(args.output_dir)

    return args


def get_event_files(root_dir: str, begin_flag=False) -> list:

    assert(os.path.exists(root_dir))
    assert(os.path.isdir(root_dir))

    frame_pattern = r"^(?:[\d]{8})-(?:[\d]{6})-(?:[\d]{6}).png$"

    png_files = []

    if begin_flag:
        it = tqdm.tqdm(os.walk(root_dir), desc=f'walk: {root_dir}')
    else:
        it = os.walk(root_dir)

    for root, dirs, filenames in it:
        for d in dirs:
            png_files.extend(get_event_files(os.path.join(root, d)))
        for f in filenames:
            m = re.match(frame_pattern, f)
            if m:
                png_files.append(os.path.join(root, f))

    return png_files

def split_by_match(png_files: list) -> dict:

    camera_pattern = r"camera(?:[-])(?:[\d]+)"
    match_pattern = r"still|match_(?:[\d]+)"
    #match_pattern = r"still"

    match_dict = defaultdict(list)

    for path in tqdm.tqdm(png_files, desc='Splitting Paths'):

        camera_name = None
        for el in os.path.normpath(path).split(os.sep):
            m = re.match(camera_pattern, el)
            if m:
                camera_name = el
                break
        if camera_name is None:
            continue

        event_name = None
        for el in os.path.normpath(path).split(os.sep):
            m = re.match(match_pattern, el)
            if m:
                event_name = el
                break
        if event_name:
            match_dict[(camera_name, event_name)].append(path)
    return match_dict


def m_do_bgr2rgb_move(camera_name: str, match_name: str, img_filepath: str, output_dir: str):


    basename = os.path.basename(img_filepath)
    output_path = os.path.join(output_dir, camera_name, match_name, 'frames', basename)

    if os.path.exists(output_path):
        return
    
    img = cv2.imread(img_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, img)


def combine_and_fix(match_dict:dict, output_dir: str) -> dict:

    assert(match_dict is not None)
    assert(len(match_dict) > 0)

    assert(os.path.exists(output_dir))
    assert(os.path.isdir(output_dir))

    match_path_pairs = list()
    for (camera_name, match_name), path_list in match_dict.items():
        os.makedirs(os.path.join(output_dir, camera_name, match_name, 'frames'), exist_ok=True)
        for path in path_list:
            match_path_pairs.append([camera_name, match_name, path])
    match_path_pairs = np.array(match_path_pairs)

    #match_path_pairs = match_path_pairs[:100, :]
    
    with Pool(8) as pool:
        new_paths = pool.starmap(m_do_bgr2rgb_move, tqdm.tqdm(zip(match_path_pairs[:,0], match_path_pairs[:,1], match_path_pairs[:,2], repeat(output_dir)), total=match_path_pairs.shape[0], desc='Run bgr2rgb'))

    new_match_dict = defaultdict(list)
    for match_name, new_path in zip(match_path_pairs[:,0], new_paths):
        new_match_dict[match_name].append(new_path)

    for (camera_name, match_name) in tqdm.tqdm(match_dict.keys(), desc='Building labels'):
        img_dir = os.path.join(output_dir, camera_name, match_name, 'frames')
        label_path = os.path.join(output_dir, camera_name, match_name, 'labels.json')
        build_config_from_directory(img_dir, label_path, force_flag=True)

    return new_match_dict


def output_match_annotations(old_match_dict: dict, new_match_dict: dict, output_dir: str):
    
    with open(os.path.join(output_dir, 'original_map.csv'), 'w') as ofile:
        writer = csv.writer(ofile)
        writer.writerow(['original_path', 'new_path'])
        for k,orig_list in tqdm.tqdm(old_match_dict.items(), desc='Trying to write matches'):
            for i,orig_path in enumerate(orig_list):
                new_list = new_match_dict.get(k)
                if new_list:
                    new_path = new_list[i]
                    writer.writerow([orig_path, new_path])



def main():

    args = get_args()

    root_dir = args.root_dir
    output_dir = args.output_dir

    png_files = get_event_files(root_dir, True)
    match_dict = split_by_match(png_files)
    new_match_dict = combine_and_fix(match_dict, output_dir)
    output_match_annotations(match_dict, new_match_dict, output_dir)




if __name__ == '__main__':
    main()

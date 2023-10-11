#!/usr/bin/env python3

import argparse
import os
import sys

import datetime
from PIL import Image

import json


def build_config_from_directory(img_dir: str, config_filepath: str, force_flag=False, merge_flag=False):

    assert(os.path.exists(img_dir))
    assert(os.path.isdir(img_dir))

    if os.path.exists(config_filepath):
        if not force_flag and not merge_flag:
            print('Error, config filepath exists: {config_filepath}', file=sys.stderr)
            exit(1)
    else:
        if merge_flag:
            print(f'Error, file does not exist for merge: {config_filepath}')
            exit(1)

    output_dict = {}

    if not merge_flag:
        output_dict = {
                'info': {
                    'year': datetime.datetime.now().year,
                    'version': "1.0",
                    'description': "custom",
                    'contributor': 'IUPUI',
                },
                'categories': [],
                'images': [],
                'annotations': [],
            }
    else:
        with open(config_filepath, 'r') as ifile:
            output_dict = json.load(ifile)
            output_dict['images'] = []

    idx = 0
    for root_path, dirs, filenames in os.walk(img_dir):

        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ext.lower() == '.png':
                img_filepath = os.path.join(root_path, f)
                img = Image.open(img_filepath)
                width, height = img.size
                entry = {
                    'id': idx,
                    'width': width,
                    'height': height,
                    'file_name': f,
                }
                output_dict['images'].append(entry)
                idx += 1

    with open(config_filepath, 'w') as ofile:
        json.dump(output_dict, ofile)


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help="Directory with images")
    parser.add_argument('--config-filepath', '-c', type=str, default='labels.json', help='Output path for config file (default: config.json)')
    parser.add_argument('--force-flag', '-f', action='store_true', help='Force config overwrite')
    parser.add_argument('--merge-flag', '-m', action='store_true', help='Merges changes from provided config file')
    return parser.parse_args()



def main():

    args = get_args()

    build_config_from_directory(args.img_dir, args.config_filepath, args.force_flag, args.merge_flag)


if __name__ == '__main__':
    main()

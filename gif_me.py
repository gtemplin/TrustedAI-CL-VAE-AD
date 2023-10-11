#!/usr/bin/env python3

import argparse, os, sys
from PIL import Image
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('img_list', nargs="+", type=str)
    parser.add_argument('--output-filename', '-o', type=str, default='output.gif')
    parser.add_argument('--period', '-p', type=int, default=500)
    return parser.parse_args()

def save_gif(img_list: list, output_filename: str, period: int):

    assert(len(img_list) > 0)

    for img_path in img_list:
        assert(os.path.exists(img_path))
        assert(os.path.isfile(img_path))

    imgs = []
    for img_path in img_list:
        print(f'Loading: {img_path}')
        imgs.append(Image.open(img_path))

    assert(len(imgs) > 0)

    print(f'Output Size: {imgs[0].size}')
    print(f'Image Mode: {imgs[0].mode}')

    output_gif = Image.new(imgs[0].mode, imgs[0].size)
    print(f'Saving to: {os.path.abspath(output_filename)}')
    output_gif.save(output_filename, format='GIF', save_all=True, append_images=imgs, duration=period, loop=0)


def main():

    args = get_args()
    save_gif(args.img_list, args.output_filename, args.period)


if __name__ == '__main__':
    main()

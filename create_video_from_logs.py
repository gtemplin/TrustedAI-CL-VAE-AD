#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import cv2
import datetime
import numpy as np
import tqdm
import sys
from PIL import Image


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-directory', '-i', required=True, type=str, help='Input Log Directory containing output directories')
    parser.add_argument('--output-path', '-o', type=str, default='recording.mkv', help='Output file path for MP4 (default=recording.mp4)')
    parser.add_argument('--frame-rate-fps', '-r', type=int, default=20, help='Frame FPS rate (default=20)')
    parser.add_argument('--force', '-f', action='store_true', help='Force overwrite')

    args = parser.parse_args()

    assert(os.path.exists(args.input_directory))
    assert(os.path.isdir(args.input_directory))
    assert(args.frame_rate_fps > 0)

    if os.path.exists(args.output_path):
        if args.force:
            assert(os.path.isfile(args.output_path))
        else:
            print(f'Error, vidoe path exists (use --force to overwrite): {args.output_path}', file=sys.stderr)
            exit(1)

    return args


def load_data_from_directory(log_directory: str):
    assert(os.path.exists(log_directory))
    assert(os.path.isdir(log_directory))

    def _build_dir(log_directory:str, minor:str):
        d = os.path.join(log_directory, minor)
        assert(os.path.exists(d))
        assert(os.path.isdir(d))
        return d

    frames_directory = _build_dir(log_directory, 'frames')
    heatmap_directory = _build_dir(log_directory, 'heatmap')
    overlay_directory = _build_dir(log_directory, 'overlay')
    err_directory = _build_dir(log_directory, 'err')
    rec_directory = _build_dir(log_directory, 'rec')

    def _get_data(d:str):
        data = {}
        for root, dirs, filenames in os.walk(d):
            for f in filenames:
                basename = os.path.basename(f)
                basename, ext = os.path.splitext(basename)
                if ext.lower() == '.png':
                    dt = datetime.datetime.strptime(basename, '%Y%m%d-%H%M%S-%f')
                    data[dt] = os.path.join(root, f)
        return data

    return {
        'frames': _get_data(frames_directory),
        'heatmap': _get_data(heatmap_directory),
        'overlay': _get_data(overlay_directory),
        'err': _get_data(err_directory),
        'rec': _get_data(rec_directory),
    }

def write_video(data:dict, output_path:str, frame_rate: int):
    
    frames_files = data['frames']
    heatmap_files = data['heatmap']
    overlay_files = data['overlay']
    err_files = data['err']

    dt_keys = set()
    for k,v in data.items():
        for k1 in v.keys():
            dt_keys.add(k1)
    sorted_dt_keys = sorted(list(dt_keys))
    print(len(sorted_dt_keys))

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out_cap = cv2.VideoWriter(output_path, fourcc, frame_rate, (1000, 800))

    try:
        fig, ((frame_ax, err_ax), (overlay_ax, heatmap_ax)) = plt.subplots(2, 2, figsize=(10,8))
        fig.suptitle('Log Playback')

        def _plot_axis(ax, f_list, dt):
            frame_filepath = f_list.get(dt)
            if frame_filepath:
                frame = cv2.imread(frame_filepath)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.clear()
                ax.imshow(frame)
                ax.axis('off')


        start_dt = sorted_dt_keys[0]
        for dt in tqdm.tqdm(sorted_dt_keys, desc='Writing video file'):
            time_delta = (dt - start_dt).total_seconds()

            fig.suptitle(f'Log Playback: {time_delta:3.2f} s')

            _plot_axis(frame_ax, frames_files, dt)
            _plot_axis(heatmap_ax, heatmap_files, dt)
            _plot_axis(overlay_ax, overlay_files, dt)
            _plot_axis(err_ax, err_files, dt)

            frame_ax.set_title('Original')
            heatmap_ax.set_title('Heatmap')
            overlay_ax.set_title('Overlay')
            err_ax.set_title('Error Image')

            plt.tight_layout()

            fig.canvas.draw()
            
            figure_arr = np.array(fig.canvas.renderer._renderer)
            figure_arr = cv2.resize(figure_arr, (1000, 800))
            img = cv2.cvtColor(figure_arr, cv2.COLOR_RGBA2BGR)
            
            out_cap.write(img)

            #plt.pause(0.05)
        print(f'Video File written to: {os.path.abspath(output_path)}')
    except Exception as e:
        raise e
    finally:

        out_cap.release()
        cv2.destroyAllWindows()


def main():

    args = get_args()

    log_directory = args.input_directory
    output_path = args.output_path
    frame_rate = args.frame_rate_fps

    data = load_data_from_directory(log_directory)
    write_video(data, output_path, frame_rate)


if __name__ == '__main__':
    main()

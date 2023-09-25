#!/usr/bin/env python3

import argparse
import os
import sys
import tqdm
import json
import datetime

import cv2

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy,
                             QMenuBar, QMenu, QOpenGLWidget, QLabel, QScrollArea, QFileDialog, QDoubleSpinBox,
                             QGridLayout, QPushButton, QMessageBox)

from PIL import Image, ImageQt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.fuzzy_vae import FuzzyVAE
from src.data_loader import load_data
from src.load_model import load_model

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


class CameraStreamerMainWindow(QMainWindow):

    def __init__(self, args):
        QMainWindow.__init__(self)

        self.cur_dir = os.path.curdir
        self.model = None
        self.config = dict()

        self.rtsp_ip = args.rtsp_ip
        self.rtsp_port = args.rtsp_port
        self.rtsp_username = args.rtsp_username
        self.rtsp_password = args.rtsp_password
        self.rtsp_url = None
        self.cap = None
        self.setup_rtsp_url()

        self.last_frame = None
        self.last_frame_qt = None
        self.last_frame_pixmap = None
        self.error_frame = None
        self.error_frame_pixmap = None

        self.update_draws_flag = False
        self.reading_frame_flag = False
        self.running_model_flag = False

        self.rec_img = None
        self.rec_pixmap = None
        self.error_img = None
        self.error_img_pixmap = None

        # Record
        self.record_dir = None
        self.record_instance_dir = None
        self.recording_flag = False
        self.handle_recording_flag = False

        self.process_rate = 0.0
        self.inference_rate_threshold = 0.25
        self.inference_prev_time = datetime.datetime.now()
        self.disable_inference_flag = False
        self.record_rate_threshold = 0.15

        #self.video_buffer_size = int(3)

        # Stream Statistics
        self.stream_error_min = float('inf')
        self.stream_error_max = -float('inf')

        self.setWindowTitle(f"Streaming: {self.rtsp_url}")

        self.build_menu()

        layout = self.build_layout()

        main_widget = QWidget()
        main_widget.setLayout(layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_draws)
        self.update_timer.start(30)

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.update_error_draws)
        self.inference_timer.start(3000)

        self.setCentralWidget(main_widget)
        self.resize(1280,480)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


    def setup_rtsp_url(self):

        rtsp_url = f"{self.rtsp_ip}:{self.rtsp_port}"
        if self.rtsp_username is not None and self.rtsp_password is not None:
            rtsp_url = f"{self.rtsp_username}:{self.rtsp_password}@{rtsp_url}"
        self.rtsp_url = f"rtsp://{rtsp_url}"
        print(f'RTSP URL: {self.rtsp_url}')
        try:

            self.cap = cv2.VideoCapture(self.rtsp_url)
            #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.video_buffer_size)

        except Exception as e:
            print(f'Failed to load RTSP: {rtsp_url}', file=sys.stderr)
            print(f'Exception: {e}', file=sys.stderr)
            self.cap = None
        

    def build_layout(self):

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        self.stream_widget = QLabel()
        self.stream_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_layout.addWidget(self.stream_widget, 1)

        self.error_label = QLabel()
        self.error_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_layout.addWidget(self.error_label, 1)

        bottom_layout = QHBoxLayout()

        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.record_btn_pressed)
        bottom_layout.addWidget(self.record_btn)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_btn_pressed)
        bottom_layout.addWidget(load_model_btn)

        self.toggle_inference_btn = QPushButton("Toggle Inference")
        self.toggle_inference_btn.clicked.connect(self.toggle_inference_btn_pressed)
        bottom_layout.addWidget(self.toggle_inference_btn)

        train_model_btn = QPushButton("Train Model")
        train_model_btn.clicked.connect(self.train_model_btn_pressed)
        bottom_layout.addWidget(train_model_btn)

        bottom_layout.addStretch()

        main_layout.addLayout(top_layout, 10)
        main_layout.addLayout(bottom_layout, 1)

        return main_layout
    
        
    def build_menu(self):

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction("Select &Camera...", self.select_camera_action)
        file_menu.addAction("Select &Record Directory...", self.select_record_dir_action)
        
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.window_exit)

        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction("Combine &Datasets...", self.combine_datasets_action)


    def select_record_dir_action(self):
        print('Select Record Directory Pressed')

        if self.recording_flag:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Error, please stop recording before switching recording directory")
            msgBox.setWindowTitle("Cannot change recording directory")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return

        selected_dir = QFileDialog.getExistingDirectory(self, "Load Recording Directory", self.record_dir, QFileDialog.ShowDirsOnly)
        print(f'Selected Directory: {selected_dir}')

        if os.path.exists(selected_dir):
            if os.path.isdir(selected_dir):
                self.record_dir = os.path.abspath(selected_dir)
                print(f'New recording dir: {self.record_dir}')


    def record_btn_pressed(self):
        print('Record Pressed')

        if self.recording_flag:
            self.terminate_recording()
            self.recording_flag = False
        else:
            self.begin_recording()


    def load_model_btn_pressed(self):
        print('Load Model Pressed')

        selected_dir = QFileDialog.getExistingDirectory(self, "Load Log Directory", self.cur_dir, QFileDialog.ShowDirsOnly)
        print(f'Selected Directory: {selected_dir}')

        if os.path.exists(selected_dir):
            if os.path.isdir(selected_dir):
                try:
                    model, config = load_model(selected_dir)
                except Exception as e:
                    print(f'Failed to load directory: {e}')
                    return
                
                self.model = model
                self.config = config
                self.cur_dir = selected_dir
                
                self.stream_error_min = float('inf')
                self.stream_error_max = -float('inf')

                #self.update_draws()
            else:
                print('Error, selected file is not a log directory')
        else:
            print(f'Error, directory does not exist')

    def combine_datasets_action(self):
        print('Combining Selected Datasets')

        selected_directories = list()
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Existing Dataset to Merge", self.cur_dir, options=QFileDialog.ShowDirsOnly)

        while os.path.isdir(selected_dir):
            selected_directories.append(selected_dir)
            selected_dir = QFileDialog.getExistingDirectory(self, "Select Existing Dataset to Merge", self.cur_dir, options=QFileDialog.ShowDirsOnly)

        if len(selected_directories) == 0:
            print('Cancelling operation')
            return

        dest_dir = QFileDialog.getExistingDirectory(self, "Select New Destination", self.cur_dir, options=QFileDialog.ShowDirsOnly)

        if not os.path.exists(dest_dir):
            print('Cancelling operation')
            return
        if not os.path.isdir(dest_dir):
            print('Cancelling operation')
            return
        
        import shutil
        from copy import deepcopy
        labels = list()

        for src_dir in selected_directories:
            
            label_filepath = os.path.join(src_dir, 'labels.json')
            if not os.path.exists(label_filepath):
                continue
            with open(label_filepath, 'r') as ifile:
                labels.append(json.load(ifile))

            for root_path, dirs, files in os.walk(src_dir):
                d_dir = root_path.replace(src_dir, dest_dir, 1)
                if not os.path.exists(d_dir):
                    os.makedirs(d_dir)
                for f in files:
                    src_file = os.path.join(root_path, f)
                    dst_file = os.path.join(d_dir, f)
                    if os.path.exists(dst_file):
                        os.remove(dst_file)
                    shutil.copy(src_file, d_dir)

        output_label = deepcopy(labels[0])
        for label_obj in labels[1:]:
            output_label['images'].extend(label_obj['images'])

        label_filepath = os.path.join(dest_dir, 'labels.json')
        with open(label_filepath, 'w') as ofile:
            json.dump(output_label, ofile)


    def toggle_inference_btn_pressed(self):
        self.disable_inference_flag = not self.disable_inference_flag
        self.toggle_inference_btn.setChecked(not self.disable_inference_flag)


    def train_model_btn_pressed(self):
        print('Train Model Pressed')

    
    def select_camera_action(self):
        print('Select Camera')


    def window_exit(self):
        print('Closing window')
        exit()

    def begin_recording(self):
        print('Recording beginning')

        if self.record_dir is None:
            print('Error, recording directory is not set')
            self.record_btn.setChecked(False)
            return
        
        if not os.path.exists(self.record_dir):
            print('Error, recording directory not set')
            self.record_btn.setChecked(False)
            return
        if not os.path.isdir(self.record_dir):
            print('Error, selected recording directory is not a directory')
            self.record_btn.setChecked(False)
            return
        
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.record_instance_dir = os.path.join(self.record_dir, f'data_{start_time}')
        os.makedirs(os.path.join(self.record_instance_dir, 'frames'))
        print(f'Recording to: {self.record_instance_dir}')
        
        self.recording_flag = True

    def terminate_recording(self):
        print('Terminating recording')
        self.recording_flag = False

        if not os.path.exists(self.record_instance_dir):
            return
        if not os.path.isdir(self.record_instance_dir):
            return
        
        img_filelist = list()

        for (dirpath, _, filenames) in os.walk(os.path.join(self.record_instance_dir, 'frames')):
            for f in filenames:
                ext = os.path.splitext(f)[1]
                if ext.lower() == '.png':
                    img_filelist.append(os.path.join(dirpath, f))

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

        for idx, img_filepath in tqdm.tqdm(enumerate(img_filelist), desc='Reading images'):
            img = Image.open(img_filepath)
            width, height = img.size
            entry = {
                'id': idx,
                'width': width,
                'height': height,
                'file_name': os.path.split(img_filepath)[1],
            }
            output_dict['images'].append(entry)

        labels_filename = os.path.join(self.record_instance_dir, 'labels.json')
        print(f'Saving labels to: {labels_filename}')
        with open(labels_filename, 'w') as ofile:
            json.dump(output_dict, ofile)





    def update_draws(self):

        if self.update_draws_flag:
            return
        self.update_draws_flag = True

        start_time = datetime.datetime.now()

        self.update_stream()
        stream_update_time = datetime.datetime.now()
        
        #self.update_error_draws()
        error_calc_draw_time = datetime.datetime.now()

        self.handle_recording()
        record_time = datetime.datetime.now()

        stream_delta = stream_update_time - start_time
        error_delta = error_calc_draw_time - stream_update_time
        record_delta = record_time - error_calc_draw_time

        process_rate = (record_time - start_time).total_seconds()
        self.process_rate = 0.9 * process_rate + 0.1 * self.process_rate

        print('*****************************')
        print(f'Process Rate: {self.process_rate}')
        print(f'Stream Delta: {stream_delta.total_seconds()}')
        print(f'Error Delta: {error_delta.total_seconds()}')
        print(f'Record Delta: {record_delta.total_seconds()}')
        print('*****************************\n')

        QApplication.processEvents()

        self.update_draws_flag = False


    def update_stream(self):

        if self.reading_frame_flag:
            return
        self.reading_frame_flag = True
        
        try:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    print(f'{time_str}: Failed to read capture devices: {self.rtsp_url}')
                    self.reading_frame_flag = False
                    return
                
                self.last_frame = frame
                self.last_frame_qt = ImageQt.ImageQt(Image.fromarray(frame))
                self.last_frame_pixmap = QPixmap.fromImage(self.last_frame_qt).copy()

                w = self.stream_widget.width()
                h = self.stream_widget.height()
                self.last_frame_pixmap = self.last_frame_pixmap.scaled(w,h, Qt.KeepAspectRatio)

                self.stream_widget.setPixmap(self.last_frame_pixmap)
                #self.stream_widget.setScaledContents(True)
                self.stream_widget.update()

                print(self.cap.get(cv2.CAP_PROP_READ_TIMEOUT_MSEC))
                #print(self.cap.get(cv2.CAP_PROP_POS_MSEC), self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
                
                #count = 1
                #while count % 5 != 0 and ret:
                #    ret = self.cap.grab()
                #    count += 1
                #    if not ret:
                #        print('No ret')
        finally:
            self.reading_frame_flag = False

    def handle_recording(self):

        if self.process_rate > self.record_rate_threshold:
            return

        if self.handle_recording_flag:
            return
        
        self.handle_recording_flag = True

        try:
            if self.recording_flag:
                if os.path.exists(self.record_instance_dir):
                    if os.path.isdir(self.record_instance_dir):
                        img = Image.fromarray(self.last_frame)
                        img_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                        img_filename = os.path.join(self.record_instance_dir, 'frames', f'{img_time}.png')
                        img.save(img_filename)
        except Exception as e:
            print(f'Failed to save image: {e}', file=sys.stderr)
        
        finally:
            self.handle_recording_flag = False


    def update_error_draws(self):

        if self.disable_inference_flag:
            return

        if self.model is None:
            return
        
        if self.process_rate > self.inference_rate_threshold:
            return

        if self.running_model_flag:
            return
        self.running_model_flag = True
        
        try:
            input_size = self.config['data']['image_size'][:2]

            #print(type(self.last_frame), tf.math.reduce_max(self.last_frame), tf.math.reduce_min(self.last_frame))

            tensor_start_time = datetime.datetime.now()

            t_img = tf.convert_to_tensor([cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)], dtype=tf.float32, )
            img = tf.image.resize(t_img, input_size, antialias=True) / 255.

            QApplication.processEvents()

            inference_start_time = datetime.datetime.now()

            r_img = self.model.call(img, False)[0]

            inference_end_time = datetime.datetime.now()

            stream_error_img = tf.reduce_sum(tf.math.pow(img[0] - r_img, 2), axis=2)
            stream_error_min = tf.reduce_min(stream_error_img)
            stream_error_max = tf.reduce_max(stream_error_img)

            self.stream_error_min = min(self.stream_error_min, stream_error_min)
            self.stream_error_max = max(self.stream_error_max, stream_error_max)

            stream_error_img = 255. * (stream_error_img - self.stream_error_min) / (self.stream_error_max - self.stream_error_min)
            stream_error_img = np.round(stream_error_img).astype(np.uint8)

            error_calculation_time = datetime.datetime.now()

            #res_img = tf.cast(tf.round(r_img * 255), dtype=tf.uint8).numpy()
            #stream_error_img_pil = Image.fromarray(res_img, mode='RGB')
            stream_error_img_pil = Image.fromarray(stream_error_img, mode='L')
            self.error_frame = ImageQt.ImageQt(stream_error_img_pil)
            self.error_frame_pixmap = QPixmap.fromImage(self.error_frame).copy()

            w = self.error_label.width()
            h = self.error_label.height()
            self.error_frame_pixmap = self.error_frame_pixmap.scaled(w, h, Qt.KeepAspectRatio)

            self.error_label.setPixmap(self.error_frame_pixmap)
            self.error_label.update()

            image_update_time = datetime.datetime.now()

            self.inference_prev_time = image_update_time

            print(f' - Tensor Conv. Delta: {(inference_start_time - tensor_start_time).total_seconds()}')
            print(f' - Inference Delta: {(inference_end_time - inference_start_time).total_seconds()}')
            print(f' - Error Calc. Delta: {(error_calculation_time - inference_end_time).total_seconds()}')
            print(f' - Image Upd. Delta: {(image_update_time - error_calculation_time).total_seconds()}')

        finally:
            self.running_model_flag = False



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rtsp_ip", type=str, help="RTSP Hostname")
    parser.add_argument("--rtsp-port", "-p", type=int, default=554, help="RTSP Port")
    parser.add_argument("--rtsp-username", "-u", type=str, default=None, help="RTSP access username")
    parser.add_argument("--rtsp-password", "-s", type=str, default=None, help="RTSP access password")

    return parser.parse_args()


if __name__ == '__main__':


    args = get_args()

    app = QApplication(sys.argv)

    main = CameraStreamerMainWindow(args)
    main.show()

    sys.exit(app.exec_())
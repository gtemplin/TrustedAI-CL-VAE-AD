#!/usr/bin/env python3

import argparse
import os
import sys
import tqdm
import json
import datetime
import time
import shutil

import cv2

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy,
                             QMenuBar, QMenu, QOpenGLWidget, QLabel, QScrollArea, QFileDialog, QDoubleSpinBox,
                             QGridLayout, QPushButton, QMessageBox, QAction, QSpinBox)

from PIL import Image, ImageQt, ImageDraw, ImageFont

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
        self.model_cache_dir = args.model_cache_dir
        self.schedule_model_save_flag = True
        self.model_changed_flag = True

        self.rtsp_ip = args.rtsp_ip
        self.rtsp_port = args.rtsp_port
        self.rtsp_username = args.rtsp_username
        self.rtsp_password = args.rtsp_password
        self.rtsp_override = args.rtsp_overide
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
        self.inf_img = None
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
        self.enable_cont_learning_flag = False

        #self.video_buffer_size = int(3)

        # Stream Statistics
        self.stream_error_min = 0.0
        self.stream_error_max = 0.0
        self.stream_error_ma = 0.99
        self.stream_error_sum_ma = 0.0
        self.stream_error_sum_2_ma = 1.0

        self.setWindowTitle(f"Streaming: {self.rtsp_url}")

        self.build_menu()

        layout = self.build_layout()

        main_widget = QWidget()
        main_widget.setLayout(layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_draws)
        self.update_timer.start(30)

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.update_inference_draws)
        #self.inference_timer.start(3000)

        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.handle_recording)
        self.record_timer.start(500)

        self.model_save_timer = QTimer()
        self.model_save_timer.timeout.connect(self.schedule_model_save)
        self.model_save_timer.start(5000)

        self.setCentralWidget(main_widget)
        self.resize(1280,480)


    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


    def setup_rtsp_url(self):

        if self.rtsp_override:
            self.rtsp_url = self.rtsp_override
        else:
            rtsp_url = f"{self.rtsp_ip}:{self.rtsp_port}"
            if self.rtsp_username is not None and self.rtsp_password is not None:
                rtsp_url = f"{self.rtsp_username}:{self.rtsp_password}@{rtsp_url}"
            self.rtsp_url = f"rtsp://{rtsp_url}"
        print(f'RTSP URL: {self.rtsp_url}')
        try:
            
            self.cap = cv2.VideoCapture(self.rtsp_url)
            #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.video_buffer_size)
            #self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FPS, 20)

        except Exception as e:
            print(f'Failed to load RTSP: {self.rtsp_url}', file=sys.stderr)
            print(f'Exception: {e}', file=sys.stderr)
            print(type(e))
            exit()
            self.cap = None

        self.negotiate_rtsp_connection()


    def negotiate_rtsp_connection(self):
        wait_period = 1

        while not self.cap.isOpened() or not self.cap.grab():
            print(f'Waiting {wait_period} seconds')
            time.sleep(wait_period)
            wait_period *= 2
            self.cap.open(self.rtsp_url)
            
        

    def build_layout(self):

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        self.stream_widget = QLabel()
        self.stream_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.stream_widget.setMinimumSize(0,0)
        top_layout.addWidget(self.stream_widget, 1)

        self.error_label = QLabel()
        self.error_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.error_label.setMinimumSize(0,0)
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
        self.toggle_inference_btn.setCheckable(True)
        self.toggle_inference_btn.setChecked(not self.disable_inference_flag)
        self.toggle_inference_btn.clicked.connect(self.toggle_inference_btn_pressed)
        bottom_layout.addWidget(self.toggle_inference_btn)

        train_model_btn = QPushButton("Train Model")
        train_model_btn.clicked.connect(self.train_model_btn_pressed)
        bottom_layout.addWidget(train_model_btn)

        self.toggle_cont_learn_btn = QPushButton("Toggle Cont. Learning")
        self.toggle_cont_learn_btn.setCheckable(True)
        self.toggle_cont_learn_btn.setChecked(self.enable_cont_learning_flag)
        self.toggle_cont_learn_btn.clicked.connect(self.toggle_cont_learn_btn_pressed)
        bottom_layout.addWidget(self.toggle_cont_learn_btn)

        lr_lbl = QLabel('LR:')
        bottom_layout.addWidget(lr_lbl)

        self.learning_rate_dsb = QDoubleSpinBox()
        self.learning_rate_dsb.setMinimum(0.0)
        self.learning_rate_dsb.setMaximum(1.0)
        self.learning_rate_dsb.setDecimals(3)
        self.learning_rate_dsb.setSingleStep(0.1)
        bottom_layout.addWidget(self.learning_rate_dsb)

        self.learning_rate_exp_sb = QSpinBox()
        self.learning_rate_exp_sb.setMinimum(-254)
        self.learning_rate_exp_sb.setMaximum(0)
        self.learning_rate_exp_sb.setSingleStep(1)
        bottom_layout.addWidget(self.learning_rate_exp_sb)

        img_noise_lbl = QLabel('Img Noise:')
        bottom_layout.addWidget(img_noise_lbl)

        self.img_noise_dsb = QDoubleSpinBox()
        self.img_noise_dsb.setMinimum(0.0)
        self.img_noise_dsb.setMaximum(1.0)
        self.img_noise_dsb.setSingleStep(0.1)
        self.img_noise_dsb.setValue(0.0)
        bottom_layout.addWidget(self.img_noise_dsb)

        self.img_noise_exp_sb = QSpinBox()
        self.img_noise_exp_sb.setMinimum(-254)
        self.img_noise_exp_sb.setMaximum(255)
        self.img_noise_exp_sb.setSingleStep(1)
        self.img_noise_exp_sb.setValue(0)
        bottom_layout.addWidget(self.img_noise_exp_sb)

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
        file_menu.addAction('&Save Model to cache', self.schedule_model_save_overide)
        
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.window_exit)

        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction("Combine &Datasets...", self.combine_datasets_action)

        view_menu = menu.addMenu("&View")
        self.show_reconstruction_action = QAction('Show Reconstructions', self, checkable=True)
        view_menu.addAction(self.show_reconstruction_action)
        self.overlay_heatmap_action = QAction('Overlay Heatmap', self, checkable=True)
        view_menu.addAction(self.overlay_heatmap_action)
        self.draw_jet_action = QAction('Draw JET', self, checkable=True)
        view_menu.addAction(self.draw_jet_action)


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

                lr = float(self.config['training']['learning_rate'])
                lr_exp = int(np.log10(lr))
                lr_man = lr / (10**lr_exp)
                self.learning_rate_dsb.setValue(lr_man)
                self.learning_rate_exp_sb.setValue(lr_exp)

                self.model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(self.config['training']['learning_rate'])
                ))

                if os.path.exists(self.model_cache_dir):
                    shutil.rmtree(self.model_cache_dir)
                os.makedirs(self.model_cache_dir)

                self.save_model_to_cache()

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

    def toggle_cont_learn_btn_pressed(self):
        self.enable_cont_learning_flag = not self.enable_cont_learning_flag
        self.toggle_cont_learn_btn.setChecked(self.enable_cont_learning_flag)


    def schedule_model_save(self):
        self.schedule_model_save_flag = True
    def schedule_model_save_overide(self):
        self.schedule_model_save_flag = True
        self.model_changed_flag = True


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
        
        self.update_inference_draws()
        error_calc_draw_time = datetime.datetime.now()

        #self.partial_fit_model()
        model_fit_time = datetime.datetime.now()

        #self.handle_recording()
        record_time = datetime.datetime.now()

        stream_delta = stream_update_time - start_time
        error_delta = error_calc_draw_time - stream_update_time
        fit_delta = model_fit_time - error_calc_draw_time
        record_delta = record_time - model_fit_time

        process_rate = (record_time - start_time).total_seconds()
        self.process_rate = 0.9 * process_rate + 0.1 * self.process_rate

        print('*****************************')
        print(f'Process Rate: {self.process_rate}')
        print(f'Stream Delta: {stream_delta.total_seconds()}')
        print(f'Error Delta: {error_delta.total_seconds()}')
        print(f'Model Fit Time: {fit_delta.total_seconds()}')
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
                    self.negotiate_rtsp_connection()
                    return
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
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


    def update_inference_draws(self):

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

            if self.inf_img is None:
                self.inf_img = tf.Variable(self.last_frame, dtype=tf.float32)
            else:
                self.inf_img.assign(value=self.last_frame)
            img = tf.image.resize(tf.expand_dims(self.inf_img, axis=0), input_size, antialias=True) / 255.

            QApplication.processEvents()

            # If Continuous Learning
            if self.enable_cont_learning_flag:

                lr_mantisa = float(self.learning_rate_dsb.value())
                lr_exp = int(self.learning_rate_exp_sb.value())
                lr = float(f'{lr_mantisa}E{lr_exp}')

                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

                img_noise_mantisa = float(self.img_noise_dsb.value())
                img_noise_exp = int(self.img_noise_exp_sb.value())
                img_noise = float(f'{img_noise_mantisa}E{img_noise_exp}')

                #tf.keras.backend.set_value(self.model.beta, img_noise)
                self.model.beta = img_noise

                #n_img = img + tf.random.normal(img.shape, 0.0, img_noise)

                loss, r_img = self.model.train_step_and_run(img)
                r_img = r_img[0]

                self.model_changed_flag = True
            else:
                r_img = self.model.call(img, False)[0]

            if self.show_reconstruction_action.isChecked():
                res_img = tf.cast(tf.round(r_img * 255.), dtype=tf.uint8).numpy()
                stream_error_img_pil = Image.fromarray(res_img, mode='RGB')
            else:

                stream_error_img = tf.reduce_sum(tf.math.pow(img[0] - r_img, 2), axis=2)
                stream_error_min = tf.reduce_min(stream_error_img)
                stream_error_max = tf.reduce_max(stream_error_img)

                self.stream_error_max = (self.stream_error_ma) * self.stream_error_max + (1.0 - self.stream_error_ma) * stream_error_max
                self.stream_error_min = (self.stream_error_ma) * self.stream_error_min + (1.0 - self.stream_error_ma) * stream_error_min

                stream_error_img_norm = (stream_error_img - self.stream_error_min) / (self.stream_error_max - self.stream_error_min)
                stream_error_img = np.round(255. * stream_error_img_norm).astype(np.uint8)
                
                stream_error_sum = tf.math.reduce_sum(stream_error_img_norm)
                self.stream_error_sum_ma = (self.stream_error_ma) * self.stream_error_sum_ma + (1. - self.stream_error_ma) * stream_error_sum
                self.stream_error_sum_2_ma = (self.stream_error_ma) * self.stream_error_sum_2_ma + (1. - self.stream_error_ma) * tf.math.pow(stream_error_sum, 2)
                stream_error_sum_var = tf.abs(self.stream_error_sum_2_ma - tf.math.pow(self.stream_error_sum_ma,2))
                anomaly_score = (stream_error_sum - self.stream_error_sum_ma) / tf.math.sqrt(stream_error_sum_var + 1E-6)

                if self.overlay_heatmap_action.isChecked():
                    heatmap = cv2.applyColorMap(stream_error_img, cv2.COLORMAP_JET)
                    print(heatmap.dtype, img.numpy().dtype)
                    superimpose = cv2.addWeighted(heatmap, 0.5, np.round(255. * img.numpy()[0]).astype(np.uint8), 0.5, 0.0)
                    stream_error_img_pil = Image.fromarray(superimpose, mode='RGB')

                else:
                    if self.draw_jet_action.isChecked():
                        heatmap = cv2.applyColorMap(stream_error_img, cv2.COLORMAP_JET)
                        stream_error_img_pil = Image.fromarray(heatmap, mode='RGB')
                    else:
                        stream_error_img_pil = Image.fromarray(stream_error_img, mode='L')

                font_y = tf.shape(r_img).numpy()[0] - 10
                drawer = ImageDraw.Draw(stream_error_img_pil)
                drawer.text((10,font_y), f'({anomaly_score:0.4g}, {stream_error_sum:0.4g}, {self.stream_error_sum_ma:0.4g}, {tf.math.sqrt(self.stream_error_sum_2_ma):0.4g})', (255,))

            self.error_frame = ImageQt.ImageQt(stream_error_img_pil)
            self.error_frame_pixmap = QPixmap.fromImage(self.error_frame).copy()

            w = self.error_label.width()
            h = self.error_label.height()
            self.error_frame_pixmap = self.error_frame_pixmap.scaled(w, h, Qt.KeepAspectRatio)

            self.error_label.setPixmap(self.error_frame_pixmap)
            self.error_label.update()

        finally:
            self.running_model_flag = False

    def save_model_to_cache(self):
        if not self.schedule_model_save_flag:
            return
        self.schedule_model_save_flag = False
        
        if not self.model_changed_flag:
            return
        
        try:
            if not os.path.exists(self.model_cache_dir):
                print(f'Error: model path does not exist: {self.model_cache_dir}', file=sys.stderr)
                return
            if not os.path.isdir(self.model_cache_dir):
                print(f'Error: model path is not a directory: {self.model_cache_dir}', file=sys.stderr)
                return

            print(f'Saving model to: {self.model_cache_dir}')

            self.model.encoder.save(os.path.join(self.model_cache_dir, 'encoder'))
            self.model.decoder.save(os.path.join(self.model_cache_dir, 'decoder'))

        except Exception as e:
            print(e)
            pass
        finally:
            self.model_changed_flag = False

        QApplication.processEvents()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rtsp_ip", type=str, help="RTSP Hostname")
    parser.add_argument("--rtsp-port", "-p", type=int, default=554, help="RTSP Port")
    parser.add_argument("--rtsp-username", "-u", type=str, default=None, help="RTSP access username")
    parser.add_argument("--rtsp-password", "-s", type=str, default=None, help="RTSP access password")
    parser.add_argument("--rtsp-overide", type=str, default=None)
    parser.add_argument("--model-cache-dir", "-d", type=str, default=os.path.join('.', '.model'))

    return parser.parse_args()


if __name__ == '__main__':


    args = get_args()

    app = QApplication(sys.argv)

    main = CameraStreamerMainWindow(args)
    main.show()

    sys.exit(app.exec_())
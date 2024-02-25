#!/usr/bin/env python3

import os
import sys
import yaml
import tqdm

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy,
                             QMenuBar, QMenu, QOpenGLWidget, QLabel, QScrollArea, QFileDialog, QDoubleSpinBox,
                             QGridLayout)

from PIL import Image, ImageQt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.load_model import load_model_from_directory

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


class ReconstructionGLWidget(QOpenGLWidget):

    def __init__(self):
        QOpenGLWidget.__init__(self)

        self.img = None

    def initializeGL(self):
        pass

    def paintGL(self):
        if self.img is not None:

            painter = QPainter()
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawImage(self.rect(), self.img)
            painter.end()

    def resizeGL(self, w:int, h:int):
        pass





class DecoderGeneratorMainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.cur_dir = os.path.curdir
        self.model = None
        self.config = dict()
        self.train_ds = None
        self.val_ds = None
        self.latent_sample = None
        self.dsb_list = None
        self.change_scheduled = False

        self.rec_img = None
        self.rec_pixmap = None
        self.hist_img = None
        self.hist_pixmap = None

        self.setWindowTitle("Decoder Image Generator v0.0")

        self.build_menu()

        layout = self.build_layout()

        main_widget = QWidget()
        main_widget.setLayout(layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_draws)
        self.update_timer.start(60) # 15 Hz

        self.setCentralWidget(main_widget)

    def build_layout(self):

        main_layout = QHBoxLayout()

        #self.rec_widget = ReconstructionGLWidget()
        self.rec_widget = QLabel()
        self.rec_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.rec_widget, 1)

        right_layout = QVBoxLayout()

        self.hist_label = QLabel()
        self.hist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.hist_label, 1)

        self.latent_scroll_area = QScrollArea()
        self.latent_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.latent_scroll_area = QWidget()
        right_layout.addWidget(self.latent_scroll_area, 1)

        main_layout.addLayout(right_layout)

        return main_layout
    
        
    def build_menu(self):

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction("&Load", self.load_menu)
        
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.window_exit)

    
    def load_config(self, config_filename: str):

        assert(os.path.exists(config_filename))
        assert(os.path.isfile(config_filename))

        # Load config file
        config = None
        try:
            with open(config_filename, 'r') as ifile:
                config = yaml.safe_load(ifile)

        except IOError as e:
            raise e
        except yaml.YAMLError as e:
            raise e

        return config 
    
    
    def load_data(self, config:dict):

        dataset_path = config['data'].get('dataset_path')
        dataset_name = config['data'].get('dataset')
        train_split = config['data']['train_split']
        val_split = config['data']['val_split']
        config_img_size = config['data']['image_size']
        img_size = (config_img_size[0], config_img_size[1])
        batch_size = config['training']['batch_size']

        if dataset_path is not None:
            print(f'Loading dataset from: {dataset_path}')
            assert(os.path.exists(dataset_path))
            
            ds = tf.data.Dataset.load(dataset_path)

            train_ds = ds.map(lambda x: x[train_split])
            val_ds = ds.map(lambda x: x[val_split])

            def normalize_img(element):
                return tf.cast(element, tf.float32) / 255.
            
            train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

        else:
            train_ds = tfds.load(dataset_name, split=train_split, shuffle_files=False)
            val_ds = tfds.load(dataset_name, split=val_split, shuffle_files=False)

            def normalize_img(element):
                return tf.cast(element['image'], tf.float32) / 255.
            
            train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        
        def resize_img(element, img_size):
            return tf.image.resize(element, size=img_size)
        
        train_ds = train_ds.map(lambda x: resize_img(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x: resize_img(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

        return train_ds, val_ds
    

    def build_latent_grid(self):

        COLS=5

        latent_vec_dim = self.latent_sample.shape[0]
        rows = np.ceil(float(latent_vec_dim) / float(COLS))

        grid_layout = QGridLayout()

        dsb_list = []

        for idx in range(latent_vec_dim):
            dsb = QDoubleSpinBox()
            dsb.valueChanged.connect(self.schedule_change)
            dsb.setRange(-100000,100000)
            dsb.setSingleStep(0.125)
            dsb.setValue(self.latent_sample[idx])
            row = idx // COLS
            col = idx % COLS
            grid_layout.addWidget(dsb, row, col)
            dsb_list.append(dsb)

        self.dsb_list = dsb_list

        widget = QWidget()
        widget.setLayout(grid_layout)
        widget.setMinimumWidth(300)
        widget.setMinimumHeight(300)
        widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        #layout = QVBoxLayout()
        #layout.addWidget(widget)
        #layout.addStretch()

        #self.latent_scroll_area.setLayout(layout)
        self.latent_scroll_area.setWidget(widget)
        self.latent_scroll_area.adjustSize()
        self.latent_scroll_area.update()



    def load_menu(self):
        print('Load button clicked')

        selected_dir = QFileDialog.getExistingDirectory(self, "Load Log Directory", self.cur_dir, QFileDialog.ShowDirsOnly)
        print(f'Selected Directory: {selected_dir}')

        if os.path.exists(selected_dir):
            if os.path.isdir(selected_dir):
                try:
                    model, config = load_model_from_directory(selected_dir)
                    train_ds, val_ds = self.load_data(config)
                except Exception as e:
                    print(f'Failed to load directory: {e}')
                    return
                
                self.model = model
                self.config = config
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.cur_dir = selected_dir

                latent_dim = int(self.config['model']['latent_dimensions'])

                self.latent_sample = np.zeros(shape=(latent_dim,))

                self.build_latent_grid()

                self.draw_hist()

                self.schedule_change()
                #self.update_draws()
            else:
                print('Error, selected file is not a log directory')
        else:
            print(f'Error, directory does not exist')


    def window_exit(self):
        print('Closing window')
        exit()


    def draw_hist(self):

        if self.train_ds is None:
            print('Error, train data is empty')
            return
        if self.val_ds is None:
            print('Error, validation data is empty')
            return
        
        if self.model is None:
            print('Error, model is empty')
            return

        z_train = []
        for batch in tqdm.tqdm(self.train_ds, desc='Z:Train'):
            _, z, _, _ = self.model.call_detailed(batch)
            z_train.append(z)
        z_train = tf.convert_to_tensor(tf.concat(z_train, axis=0))
        print(z_train.shape)

        z_val = []
        for batch in tqdm.tqdm(self.val_ds, desc='Z:Val'):
            _, z, _, _ = self.model.call_detailed(batch)
            z_val.append(z)
        z_val = tf.convert_to_tensor(tf.concat(z_val, axis=0))
        print(z_val.shape)

        fig, ax = plt.subplots(1, 1)
        ax.hist(tf.reshape(z_train, [-1]).numpy(), bins=64, density=True, label='train', alpha=0.65)
        ax.hist(tf.reshape(z_val, [-1]).numpy(), bins=64, density=True, label='val', alpha=0.65)
        ax.set_title('Latent Vector Histogram')
        ax.grid()
        ax.legend()
        ax.margins(0)

        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        img = Image.fromarray(image_from_plot, mode='RGB')
        self.hist_img = ImageQt.ImageQt(img)

        self.hist_pixmap = QPixmap.fromImage(self.hist_img).copy()

        self.hist_label.setPixmap(self.hist_pixmap)
        self.hist_label.update()

    
    def schedule_change(self):
        self.change_scheduled = True


    def update_draws(self):
        if not self.change_scheduled:
            return
        else:
            self.change_scheduled=False

            if self.model is None:
                return
            if self.latent_sample is None:
                return
            
            self.update_latent_sample()
            self.draw_reconstruction_from_sample()
            
            self.update()


    def update_latent_sample(self):

        if self.latent_sample is None:
            return
        
        if self.dsb_list is None:
            return
        
        assert(len(self.latent_sample) == len(self.dsb_list))

        for idx in range(len(self.dsb_list)):
            self.latent_sample[idx] = self.dsb_list[idx].value()

    def draw_reconstruction_from_sample(self):

        if self.model is None:
            return
        if self.latent_sample is None:
            return
        
        r_img = self.model.decode(self.latent_sample.reshape((1, -1)), True)[0]
        r_img = np.round(r_img * 255.).astype(np.uint8)

        fig, ax = plt.subplots(1,1)
        ax.imshow(r_img)
        ax.margins(0)
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        img = Image.fromarray(image_from_plot, mode='RGB')
        self.rec_img = ImageQt.ImageQt(img)

        self.rec_pixmap = QPixmap.fromImage(self.rec_img).copy()

        self.rec_widget.setPixmap(self.rec_pixmap)
        self.rec_widget.update()



if __name__ == '__main__':

    app = QApplication(sys.argv)

    main = DecoderGeneratorMainWindow()
    main.show()

    sys.exit(app.exec_())
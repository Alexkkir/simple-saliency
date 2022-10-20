import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import skvideo.io
from pathlib import Path
import os
import csv
from time import time
from threading import BoundedSemaphore, Thread
import traceback
import math
import pathlib
import json
from copy import deepcopy
import pickle as pkl
import logging




class VideoReader:
    def __init__(self, path, shape=None, batch_size=10, verbose=False):
        if str(path).endswith('yuv'):
            self.vr = skvideo.io.vreader(path, 1080, 1920, inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            self.vr = skvideo.io.vreader(path)
        self.batch_size = batch_size
        self.shape = shape or tuple(int(x) for x in tuple(torch.Tensor(skvideo.io.FFmpegReader(path).getShape())))
        self.len = self.shape[0]
        self.pos = 0
        self.time_out = 10
        self.verbose = verbose
        
    def __iter__(self):
        for batch_num in range(0, self.len, self.batch_size):
            if self.verbose and batch_num % self.time_out == 0:
                print(f'batch {batch_num // self.batch_size} / {self.len // self.batch_size}')
                
            batch_size = min(self.batch_size, self.len - self.pos)
            shape = (batch_size, *self.shape[1:])
            
            batch = np.zeros(shape)
            
            for i, frame in enumerate(self.vr):
                if i < batch_size:
                    if frame.shape != self.shape[1:]:
                        frame = cv2.resize(frame, (shape[2], shape[1]))
                    batch[i, ...] = frame / 255
                else:
                    break
            batch = torch.Tensor(batch).permute(0, 3, 1, 2)
            yield batch
            
class MyWriter:
    def __init__(self, f):
        self.filename = f
        if not os.path.exists(self.filename):
            os.open(self.filename, os.O_CREAT)

    def write_row(self, row):
        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def write_text(self, text):
        with open(self.filename, 'a') as f:
            f.write(text)
            print(text)
            
def folder2name(folder):
    name = os.path.basename(os.path.normpath(folder))
    name = "_".join(name.split("_")[:-1])
    return name

class CommandHandler:
    def __init__(self, max_processes=2):
        self.maxp = max_processes
        self.sem = BoundedSemaphore(self.maxp)
    def run(self, target, args):
        def func_with_end(args):
            target(*args)
            self.sem.release()
            # print('end')
            
        self.sem.acquire()
        thread = Thread(target=func_with_end, args=(args,))        
        thread.start()
        
def listdir(dir_name: Path):
    return sorted([dir_name / file for file in os.listdir(dir_name) if '.ipynb' not in str(file)])

def json_representation_of_dataset(dataset_path):
    json_data = {}
    for sequence_folder in os.listdir(dataset_path / 'seq'):
        sequence_name = sequence_folder.removesuffix('_x265')
        
        reference = [str(file) for file in listdir(dataset_path / 'ref') if sequence_name in str(file)][0]
        mask = [str(file) for file in listdir(dataset_path / 'masks') if sequence_name in str(file)][0]
        distorted = [str(file) for file in listdir(dataset_path / 'seq' / sequence_folder)]
        
        json_data[sequence_name] = {
            'mask': mask,
            'ref': reference,
            'dis': distorted
        }
        
    return json_data


def safe_mean(arr):
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    val = float(arr.mean())
    return val

def get_logger(log_filepath, info_filepath, errors_filepath):
    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)-7s || %(asctime)s.%(msecs)03d || %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_filepath, 'a+')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    fh_info = logging.FileHandler(info_filepath, 'a+')
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(formatter)
    
    fh_err = logging.FileHandler(errors_filepath, 'a+')
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fh_err)
    logger.addHandler(fh_info)
    return logger
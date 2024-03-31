from genericpath import isdir
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pandas as pd
import os
import cv2
from skimage import io
import numpy as np
import pickle
import yaml
from cerberus import Validator
import logging
import sys
from PIL import Image
import math
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
import joblib

def read_image(img_path):
    img = Image.open(img_path)

    return img
    
class ImageDataset(Dataset):

    def __init__(self, img_dir, class_names, transform_main=lambda x: x, transform_augment=lambda x: x):
        self.img_dir = img_dir
        self.transform_main = transform_main
        self.transform_augment = transform_augment
        self.class_names = class_names
        
        meta_path = f"metadata/{img_dir.replace('../', '')}"

        if os.path.exists(meta_path):
            logging.info(f"Using cached metadata from '{meta_path}'.")
            self.load_meta_data(meta_path)
        else:
            logging.info(f"No metadata present in'{meta_path}'.")
            logging.info("Calculating metadata...")
            
            self.calculate_metadata(img_dir)

            logging.info(f"Creating metadata in '{meta_path}'.")
            self.save_meta_data(meta_path)

    def calculate_metadata(self, img_dir):
        self.img_name_to_label = dict()
        
        self.img_names = list()

        self.label_to_classname = dict()

        self.features = dict()

        for class_idx, class_name in enumerate(self.class_names):
            self.label_to_classname[class_idx] = class_name

            for f in sorted(os.listdir(f"{img_dir}/{class_name}")):
                name = f"{class_name}/{f}"
                label = class_idx

                self.img_names.append(name)

                self.img_name_to_label[name] = label


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx] 
        img_path = f"{self.img_dir}/{img_name}"

        image = read_image(img_path)
        image = self.transform_main(image)
        image = self.transform_augment(image)
        
        label = self.img_name_to_label[img_name]
        
        return img_name, image, label

    def save_meta_data(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/img_to_label.pkl", "wb") as f:
            pickle.dump(self.img_name_to_label, f)

        with open(f"{path}/label_to_class.pkl", "wb") as f:
            pickle.dump(self.label_to_classname, f)

        with open(f"{path}/features.pkl", "wb") as f:
            pickle.dump(self.features, f)
        
        with open(f"{path}/img_names.pkl", "wb") as f:
            pickle.dump(self.img_names, f)

    def load_meta_data(self, path):
        with open(f"{path}/img_to_label.pkl", "rb") as f:
            self.img_name_to_label = pickle.load(f)

        with open(f"{path}/label_to_class.pkl", "rb") as f:
            self.label_to_classname = pickle.load(f)

        with open(f"{path}/features.pkl", "rb") as f:
            self.features = pickle.load(f)
        
        with open(f"{path}/img_names.pkl", "rb") as f:
            self.img_names = pickle.load(f)
import math
import random
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import zipfile
import os
from pathlib import Path
import pandas as pd

from utils.Model2_LSTM import Model2_LSTM
from utils.Model3_ViT import Model3_ViT
from utils.Croper import Croper
from utils.Data import ImagesDataset
from utils.Config import config

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import timm
import pydicom

def load_lstm_model(device):
    lstm_model_path = "./utils/best_model_LSTM_fold_0.pt"

    backbone = timm.create_model('efficientnet_b3.ra2_in1k', pretrained=True)
    lstm_model = Model2_LSTM(backbone=backbone, config=config)  # 实例化 Model2_LSTM
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device), strict=False)
    lstm_model.to(device)
    lstm_model.eval() 

    return lstm_model

def load_vit_model(device):
    Transformer_checkpoint = torch.load("./utils/pytorch_model_Transformer.bin")

    backbone = timm.create_model('resnet50', pretrained=True)
    model_ViT = Model3_ViT(backbone=backbone, config=config)
    model_ViT.load_state_dict(Transformer_checkpoint)
    model_ViT = model_ViT.to(device)
    model_ViT.eval()

    return model_ViT

def add_blue_box_to_image(image):
    """
    在给定的图像上添加蓝色框并返回带框的图像。
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    # 设置蓝色框的位置和大小
    width, height = image.size
    box_x = width * 0.607  # 框的位置（右边部分）
    box_y = height * 0.495
    box_width = width * 0.055
    box_height = height * 0.125

    # 添加蓝色框
    rect = patches.Rectangle((box_x, box_y), box_width, box_height,
                             linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    # 去掉坐标轴
    ax.axis("off")

    # 将带框的图像转换回 PIL 格式
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    marked_image = Image.open(buf)
    plt.close(fig)

    return marked_image

class Classifer():

    def __init__(self, crop_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lstm = load_lstm_model(self.device)
        self.model_vit = load_vit_model(self.device)

        self.croper = Croper(crop_model_path)
        self.imageset = ImagesDataset()
        

    def process_folder(self, file_path):
        resuls = {}

        # Condition labels
        conditions = [
            'left_neural_foraminal_narrowing_l1_l2',
            'left_neural_foraminal_narrowing_l2_l3',
            'left_neural_foraminal_narrowing_l3_l4',
            'left_neural_foraminal_narrowing_l4_l5',
            'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2',
            'right_neural_foraminal_narrowing_l2_l3',
            'right_neural_foraminal_narrowing_l3_l4',
            'right_neural_foraminal_narrowing_l4_l5',
            'right_neural_foraminal_narrowing_l5_s1'
        ]

        # Severity labels corresponding to each integer value
        severity_labels = ["Normal/Mild", "Moderate", "Severe"]

        file_path = Path(file_path)
        inputs, images = self.imageset.load_data(file_path, self.croper)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'labels'}

        _, outputs_lstm = self.model_lstm(**inputs)
        preds_lstm = torch.argmax(outputs_lstm, dim=2).cpu().numpy()
        df_lstm = pd.DataFrame({
            'condition': conditions,
            'severity': [severity_labels[severity] for severity in preds_lstm[0]]
        })
        resuls['lstm'] = df_lstm

        _, outputs_vit = self.model_vit(**inputs)
        preds_vit = torch.argmax(outputs_vit, dim=2).cpu().numpy()
        df_vit = pd.DataFrame({
            'condition': conditions,
            'severity': [severity_labels[severity] for severity in preds_vit[0]]
        })
        resuls['vit'] = df_vit

        outputs_ensemble = (outputs_lstm + outputs_vit) / 2
        preds_ensemble = torch.argmax(outputs_ensemble, dim=2).cpu().numpy()
        df_ensemble = pd.DataFrame({
            'condition': conditions,
            'severity': [severity_labels[severity] for severity in preds_ensemble[0]]
        })
        resuls['ensemble'] = df_ensemble

        st.session_state.predictions = resuls

        st.session_state.folder_images = images
        
        return st.session_state.predictions

classifier = Classifer("./utils/resnet18_1_Input3Channels.pt")

if __name__ == "__main__":
    
    classifier.process_folder(Path("D:/files/nus/courses/ISY5002/4-practice_module/project/test_images/13317052"),)
    print("Done")
        
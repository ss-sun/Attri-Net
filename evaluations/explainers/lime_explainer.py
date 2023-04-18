import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torchvision.models import resnet50
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from tqdm import tqdm
from train_utils import to_numpy

"""
Code adapted from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
"""

def convert_to_1channel(images, img_size):
    # Convert 3 channel to 1 channel, because models are trained on grey scale chest x-ray images
    images_1channel = []
    for idx in range(len(images)):
        img = np.resize(images[idx], (3, img_size, img_size))
        img = np.mean(img, axis=0)
        new_img = np.mean(img, axis=0, keepdims=True)
        images_1channel.append(new_img)
    return images_1channel


def get_preprocess_transform():
    # Transforms for the perturbed image, the perturbed image will be feed into the neural network model.
    transf = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transf


class lime_explainer():

    def __init__(self, model, labels):
        self.explainer = lime_image.LimeImageExplainer()
        self.model = model
        self.labels = labels

    def get_attributions(self, input, target_label_idx, positive_only):
        explanation = self.explainer.explain_instance(to_numpy(input.squeeze()),
                                                 self.batch_predict,  # Classification function
                                                 labels = self.labels, #labels=(0, 1, 2, 3, 4, 5),
                                                 top_labels=None,
                                                 hide_color=0,
                                                 num_samples=100)
        temp, mask = explanation.get_image_and_mask(label=target_label_idx, positive_only=positive_only, num_features=5, hide_rest=True)
        return mask

    def batch_predict(self, images):
        preprocess_transform = get_preprocess_transform()
        images_1channel = convert_to_1channel(images, img_size=320)
        self.model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images_1channel), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.float()
        batch = batch.to(device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
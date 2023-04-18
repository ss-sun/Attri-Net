import os
import argparse
import torch
from tqdm import tqdm
from torchvision import models, transforms
import torchvision
import numpy as np
from captum.attr import IntegratedGradients, GuidedBackprop, InputXGradient, Saliency, LayerGradCam, DeepLift, LayerAttribution
from torch import nn as nn
import shap


# Adapt from https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20MobilenetV2%20using%20the%20Partition%20explainer%20%28PyTorch%29.html

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 1 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 1 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:

    if x.dim() == 4:
        x = x if x.shape[3] == 1 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 1 else x.permute(1, 2, 0)
    return x


class shap_explainer():

    def __init__(self, model, labels):
        # Define a masker that is used to mask out partitions of the input image.
        masker_blur = shap.maskers.Image("blur(320,320)", (320, 320, 1))
        self.explainer = shap.Explainer(self.predict, masker_blur, output_names=labels)
        self.model = model
        self.labels = labels
        self.img_size = 320
        self.transform, self.inv_transform = self.create_transforms()
        self.shap_num_eval =1000
        self.shap_batchsize = 50


    def get_attributions(self, input, target_label_idx, positive_only):
        shap_values = self.explainer(input, max_evals=self.shap_num_eval, batch_size=self.shap_batchsize)
        shap_values.data = self.inv_transform(shap_values.data).cpu().numpy()[0]  # equal to the input data
        shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]
        attr = shap_values.values[target_label_idx].squeeze()
        if positive_only:
            attr = attr.clip(0)
        return attr



    def create_transforms(self):
        transform = [
            transforms.Lambda(nhwc_to_nchw),
            transforms.Lambda(lambda x: x * (1 / 255)),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Lambda(nchw_to_nhwc),
        ]
        inv_transform = [
            torchvision.transforms.Lambda(nhwc_to_nchw),
            torchvision.transforms.Lambda(nchw_to_nhwc),
        ]
        transform = torchvision.transforms.Compose(transform)
        inv_transform = torchvision.transforms.Compose(inv_transform)
        return transform, inv_transform


    def predict(self, img: np.ndarray) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = nhwc_to_nchw(torch.Tensor(img))
        img = img.to(device)
        output = self.model(img)
        return output


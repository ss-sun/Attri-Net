from __future__ import print_function
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import wandb
from train_utils import to_numpy
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from evaluations.explainers.captum_explainers import GCam_explainer, GB_explainer
from evaluations.explainers.lime_explainer import lime_explainer
from evaluations.explainers.shap_explainer import shap_explainer
from evaluations.explainers.gif_explainer import gif_explainer

class resnet_solver(object):
    # Train and test ResNet50.
    def __init__(self, exp_configs, data_loader):
        self.exp_configs = exp_configs
        self.TRAIN_DISEASES = exp_configs.train_diseases
        self.use_gpu = exp_configs.use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if exp_configs.mode == "train":
            self.img_size = exp_configs.image_size
            self.epochs = exp_configs.epochs
            self.lr = exp_configs.lr
            self.weight_decay = exp_configs.weight_decay
            self.ckpt_dir = exp_configs.ckpt_dir
            self.train_loader = data_loader['train']
            self.valid_loader = data_loader['valid']
            self.test_loader = data_loader['test']
            # Initialize model.
            self.model = self.init_model()
            self.loss = torch.nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if exp_configs.mode == "test":
            self.model_path = exp_configs.model_path
            self.valid_loader = data_loader['valid']
            self.test_loader = data_loader['test']
            self.model = self.init_model()
            self.load_model(self.model_path)



    def init_model(self):
        # Prepare model, input dim=1 because input x-ray images are gray scale.
        model = resnet50(pretrained=False, num_classes=len(self.TRAIN_DISEASES))
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(self.device)
        return model

    def load_model(self, model_path):
        model_dir = model_path + "/ckpt"
        model_path = os.path.join(model_dir, 'best_classifier.pth')
        self.model.load_state_dict(torch.load(model_path))


    def train(self):
        max_batches = len(self.valid_loader)
        best_valid_auc = 0.0
        self.model.train()
        for epoch in range(self.epochs):
            steps = 0
            train_epoch_loss = 0.0
            for idx, data in enumerate(self.train_loader):
                train_data = data['img']
                train_labels = data['label']
                train_data = train_data.to(self.device)
                train_labels = train_labels.to(self.device)
                y_pred = self.model(train_data)
                train_loss = self.loss(y_pred, train_labels)
                train_epoch_loss += train_loss.item()
                wandb.log({"train_step_loss": train_loss})
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                steps += 1
            train_epoch_loss = train_epoch_loss / steps
            wandb.log({"train_epoch_loss": train_epoch_loss})

            # Validation
            valid_auc_mean, _, _ = self.validation(max_batches=max_batches)
            # Save best model
            if best_valid_auc <= valid_auc_mean:
                best_valid_auc = valid_auc_mean
                wandb.log({"best_valid_auc": best_valid_auc})
                torch.save(self.model.state_dict(), '{:}/best_classifier.pth'.format(self.ckpt_dir))
            print('Epoch=%s, valid_AUC=%.4f, Best_valid_AUC=%.4f' % (epoch, valid_auc_mean, best_valid_auc))
            self.model.train()


    def validation(self, max_batches):
        self.model.eval()
        with torch.no_grad():
            valid_pred = []
            valid_true = []
            valid_epoch_loss = 0.0
            steps = 0

            for jdx, data in enumerate(self.valid_loader):
                if jdx >= max_batches:
                    break
                valid_data = data['img']
                valid_labels = data['label']
                valid_data = valid_data.to(self.device)
                valid_labels = valid_labels.to(self.device)
                y_pred_logits = self.model(valid_data)
                y_pred = torch.sigmoid(y_pred_logits)
                valid_loss = self.loss(y_pred_logits, valid_labels)
                valid_epoch_loss += valid_loss.item()
                wandb.log({"valid_step_loss": valid_loss})
                valid_pred.append(to_numpy(y_pred))
                valid_true.append(to_numpy(valid_labels))
                steps += 1

            valid_epoch_loss = valid_epoch_loss / steps
            wandb.log({"valid_epoch_loss": valid_epoch_loss})

            valid_true = np.concatenate(valid_true)
            valid_pred = np.concatenate(valid_pred)
            valid_auc_mean = roc_auc_score(valid_true, valid_pred)
            wandb.log({"valid_auc_mean": valid_auc_mean})

            for i in range(len(self.TRAIN_DISEASES)):
                t = valid_true[:, i]
                p = valid_pred[:, i]
                auc = roc_auc_score(t, p)
                print(self.TRAIN_DISEASES[i] + ": " + str(auc))
                wandb.log({self.TRAIN_DISEASES[i]: auc})

        return valid_auc_mean, valid_true, valid_pred


    def test(self, which_loader="valid", save_result=False, result_dir=None):

        if which_loader == "test":
            data_loader = self.test_loader
        if which_loader == "valid":
            data_loader = self.valid_loader

        self.model.eval()
        with torch.no_grad():
            test_pred = []
            test_true = []
            diter = iter(data_loader)
            for i in tqdm(range(len(data_loader))):
                data = diter.next()
                test_data = data['img']
                test_labels = data['label']
                test_data = test_data.to(self.device)
                test_labels = test_labels.to(self.device)
                y_pred_logits = self.model(test_data)
                y_pred = torch.sigmoid(y_pred_logits)
                test_pred.append(to_numpy(y_pred))
                test_true.append(to_numpy(test_labels))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_auc_mean = roc_auc_score(test_true, test_pred)
            print('test_auc_mean: ', test_auc_mean)

            for i in range(len(self.TRAIN_DISEASES)):
                t = test_true[:, i]
                p = test_pred[:, i]
                auc = roc_auc_score(t, p)
                print(self.TRAIN_DISEASES[i] + " auc: " + str(auc))

            pred = np.asarray(test_pred)
            true = np.asarray(test_true)

            if save_result:
                pred_path = os.path.join(result_dir, which_loader + '_pred.txt')
                true_path = os.path.join(result_dir, which_loader + '_true.txt')
                np.savetxt(pred_path, pred)
                np.savetxt(true_path, true)

        return test_pred, test_true, test_auc_mean




    def get_optimal_thresholds(self, save_result=True, result_dir=None):

        pred, true, valid_auc = self.test(which_loader="valid", save_result=True, result_dir=result_dir)
        print("validation auc: ", valid_auc)
        best_threshold = {}
        threshold = []

        for i in range(len(self.TRAIN_DISEASES)):
            disease = self.TRAIN_DISEASES[i]
            statics = {}
            p = pred[:, i]
            t = true[:, i]
            statics['fpr'], statics['tpr'], statics['threshold'] = roc_curve(t, p, pos_label=1)
            sensitivity = statics['tpr']
            specificity = 1 - statics['fpr']
            sum = sensitivity + specificity  # this is Youden index
            best_t = statics['threshold'][np.argmax(sum)]
            best_threshold[disease] = best_t
            threshold.append(best_t)
        threshold = np.asarray(threshold)
        print(best_threshold)

        if save_result:
            path = os.path.join(result_dir, 'best_threshold.txt')
            np.savetxt(path, threshold)
        return best_threshold





    def set_explainer(self, which_explainer):
        labels = np.arange(len(self.TRAIN_DISEASES))
        if which_explainer == 'lime':
            self.explainer = lime_explainer(self.model, labels)

        if which_explainer == 'shap':
            self.explainer = shap_explainer(self.model, labels)

        if which_explainer == 'GCam':
            self.explainer = GCam_explainer(self.model, labels)

        if which_explainer == 'GB':
            self.explainer = GB_explainer(self.model, labels)

        if which_explainer == 'gifsplanation':
            self.explainer = gif_explainer(self.model, labels)



    def get_attributes(self, img, label_idx, positive_only=True):
        """
        Get the attributions of the image for the given label index.
        """
        img = img.to(self.device)
        attr_maps = self.explainer.get_attributions(input=img, target_label_idx=label_idx, positive_only=positive_only)
        return attr_maps


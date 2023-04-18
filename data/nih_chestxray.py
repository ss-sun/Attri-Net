import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.data_utils import map_image_to_intensity_range, normalize_image


class NIHChestXray(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms, img_size, with_BBox=False):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms
        self.img_size = img_size
        self.with_BBox = with_BBox
        if self.with_BBox:
            self.image_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_BB/NIHChestX-rays/images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Image Index'])
        img = Image.open(img_path).convert("L")

        scale_factor_x = self.img_size / float(img.size[0])
        scale_factor_y = self.img_size / float(img.size[1])

        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

        data['img'] = img
        if self.with_BBox:
            label = np.zeros(len(self.TRAIN_DISEASES))
            bbox = np.zeros((len(self.TRAIN_DISEASES), 4))
            img_id = self.df.iloc[idx]['Image Index']

            rows = self.df.loc[self.df['Image Index'] == img_id]
            for index, row in rows.iterrows():
                lesion_type = row['Finding Label']
                if lesion_type in self.TRAIN_DISEASES:
                    idx = self.TRAIN_DISEASES.index(lesion_type)
                    label[idx] = 1
                    x_min = int(row['Bbox [x'] * scale_factor_x)
                    y_min = int(row['y'] * scale_factor_y)
                    width = int(row['w'] * scale_factor_x)
                    height = int(row['h]'] * scale_factor_y)
                    bbox[idx] = np.array([x_min, y_min, width, height])

            data['label'] = label
            data['BBox'] = bbox


        else:
            # Get labels from the dataframe for current image
            label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
            label = np.array(label)
            data['label'] = label

        return data





class NIHChestXrayDataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=True, img_size=320, seed=42):

        self.image_dir = dataset_params["image_dir"]
        self.data_entry_csv_file = dataset_params["data_entry_csv_file"]
        self.BBox_csv_file = dataset_params["BBox_csv_file"]
        self.train_valid_list_file = dataset_params["train_valid_list_file"]
        self.test_list_file = dataset_params["test_list_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]

        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.split_df_dir = os.path.join(os.path.dirname(self.data_entry_csv_file), 'split_df')
        self.pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                            'Infiltration', 'Mass', 'No Finding', 'Nodule',
                            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

        # get official split train_val image list
        with open(self.train_valid_list_file) as f:
            self.train_valid_list = f.read().splitlines()
        # get official split test image list
        with open(self.test_list_file) as f:
            self.test_list = f.read().splitlines()

        self.data_transforms = {
            'train': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.ToTensor(),
            ]),
            'test': tfs.Compose([
                tfs.Resize((self.img_size, self.img_size)),
                tfs.ToTensor(),
            ]),
        }

    def setup(self):

        # store split dataframe folder
        if os.path.exists(self.split_df_dir) and self.resplit==False:
            # Directly read splitted df.
            print('Already split data, will use previous created splitting dataframe!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train_df.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid_df.csv'))
            self.test_df = pd.read_csv(os.path.join(self.split_df_dir, 'test_df.csv'))

        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            # Split the train_val list to train_list and valid_list.
            self.train_list, self.valid_list = self.split(self.train_valid_list, self.split_ratio, self.seed)
            # Read the data_entry and create new dataframe that have 16 columns correspoding to the 15 diseases, and image_id column.
            new_df = self.restructure_csv()
            self.train_df, self.valid_df, self.test_df = self.create_df(new_df)

        self.BBox_test_df = pd.read_csv(self.BBox_csv_file)
        self.train_set = NIHChestXray(image_dir=self.image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'], img_size=self.img_size, with_BBox=False)
        self.valid_set = NIHChestXray(image_dir=self.image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.test_set = NIHChestXray(image_dir=self.image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.BBox_test_set = NIHChestXray(image_dir=self.image_dir, df=self.BBox_test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=True)

        # To train MT_VAGAN, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()



    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def test_dataloader(self, concat_testset=False, batch_size=1, shuffle=False, drop_last=True):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)

    def BBox_test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.BBox_test_set, batch_size=batch_size, shuffle=shuffle)

    def split(self, train_valid_list, train_ratio, seed, shuffle=True):
        df = pd.read_csv(self.data_entry_csv_file)
        patient_ids = df['Patient ID'].tolist()
        unique_patient = np.unique(np.asarray(patient_ids))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(unique_patient)

        split = int(np.floor(train_ratio * len(unique_patient)))
        train_patientID, valid_patientID = unique_patient[:split], unique_patient[split:]

        print("number of patient for training: ", len(train_patientID))
        print("number of patient for validation: ", len(valid_patientID))
        train_indices = []
        valid_indices = []

        for index, row in df.iterrows():
            patient_id = row['Patient ID']
            img_id = row["Image Index"]
            if patient_id in train_patientID and img_id in train_valid_list:
                train_indices.append(index)
            if patient_id in valid_patientID and img_id in train_valid_list:
                valid_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        train_list = train_df["Image Index"].tolist()
        valid_list = valid_df["Image Index"].tolist()

        return np.asarray(train_list), np.asarray(valid_list)


    def restructure_csv(self):
        """
        Restructure the original data_entry csv files. columns are 28 labels + image_id
        """
        df = pd.read_csv(self.data_entry_csv_file)
        img_ids = df['Image Index'].tolist()
        # get all pathologies
        df_labels = df['Finding Labels'].tolist()
        labels = set()
        for i in range(len(df_labels)):
            label = df_labels[i].split('|')
            labels.update(label)
        self.pathologies = sorted(labels)
        print("All pathologies: ", self.pathologies)

        # create new dataframe
        new_df = pd.DataFrame(0.0, index=np.arange(len(df)), columns=self.pathologies)
        new_df['Image Index'] = img_ids
        for i, row in new_df.iterrows():
            img_id = row['Image Index']
            r = df.loc[df['Image Index'] == img_id]
            finding_labels = r['Finding Labels'].tolist()[0].split('|')
            for label in finding_labels:
                row[label] = 1.0
                new_df.loc[i] = row
        return new_df


    def create_df(self, new_df):

        train_df = new_df[new_df['Image Index'].isin(self.train_list)]
        valid_df = new_df[new_df['Image Index'].isin(self.valid_list)]
        test_df = new_df[new_df['Image Index'].isin(self.test_list)]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.split_df_dir, 'train_df.csv'))
        valid_df.to_csv(os.path.join(self.split_df_dir, 'valid_df.csv'))
        test_df.to_csv(os.path.join(self.split_df_dir, 'test_df.csv'))

        return train_df, valid_df, test_df


    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size, shuffle=shuffle)
            train_dataloaders[c] = train_loader
        return train_dataloaders


    def single_disease_vis_dataloaders(self, batch_size, shuffle=False):
        vis_dataloaders = {}
        for c in ['neg', 'pos']:
            vis_loader = {}
            for disease in self.TRAIN_DISEASES:
                vis_loader[disease] = DataLoader(self.single_disease_vis_sets[c][disease], batch_size=batch_size, shuffle=shuffle)
            vis_dataloaders[c] = vis_loader
        return vis_dataloaders


    def create_trainsets(self):
        """
        create positive trainset and negative trainset for each disease
        """
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_df=self.train_df, disease=disease, label=c, transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets


    def create_vissets(self):
        """
        create positive and negative visualization set for each disease
        """
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_df=self.train_df[0:2000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_df, disease, label, transforms):
        """
        create subset from source dataset using given selected indices.
        """
        if label == 'pos':
            idx = np.where(src_df[disease] == 1)[0]
        if label == 'neg':
            idx = np.where(src_df[disease] == 0)[0]
        filtered_df = src_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = NIHChestXray(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms, img_size=self.img_size, with_BBox=False)
        return subset















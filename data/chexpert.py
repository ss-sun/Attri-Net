import os
import shutil
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import normalize_image, map_image_to_intensity_range


class CheXpert(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = {}
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Path'])
        img = Image.open(img_path).convert("L")

        if self.transforms is not None:
            img = self.transforms(img)  # return image in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

        # Get labels from the dataframe for current image
        label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
        label = np.array(label)
        data['img'] = img
        data['label'] = label
        return data





class CheXpertDataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=True, img_size=320, seed=42):

        self.image_dir = dataset_params["image_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.test_csv_file = dataset_params["test_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.orientation = dataset_params["orientation"]
        self.uncertainty = dataset_params["uncertainty"]
        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.split_df_dir = os.path.join(self.image_dir, 'split_df')
        self.data_transforms = {
            'train': tfs.Compose([
                tfs.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                tfs.Resize((self.img_size, self.img_size)),
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
            # directly read splitted data frame
            print('Already split data. Will use the previously created split data frame!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train_df.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid_df.csv'))
            self.test_df = pd.read_csv(os.path.join(self.split_df_dir, 'test_df.csv'))

        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            # first read train and test csv files
            train_df = self.preprocess_df(self.train_csv_file)
            # deal with nan and uncertainty labels
            train_df = self.fillnan_approach(train_df)
            train_df = self.uncertainty_approach(train_df)
            # split the train dataframe into train and valid dataframe, and save to csv file
            self.train_df, self.valid_df, self.test_df = self.split(df=train_df, train_ratio=self.split_ratio, seed=self.seed, shuffle=True)

        self.official_vaild_df = self.preprocess_df(self.test_csv_file)
        self.train_set = CheXpert(image_dir=self.image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'])
        self.valid_set = CheXpert(image_dir=self.image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.test_set = CheXpert(image_dir=self.image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])
        self.official_vaild_set = CheXpert(image_dir=self.image_dir, df=self.official_vaild_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'])

        # To train Attri-Net, we need to get pos_dataloader and neg_dataloader for each disease.
        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()

        self.vis_healthy = self.create_vis_healthy(transforms=self.data_transforms['test'])
        self.vis_anomaly = self.create_vis_anomaly(transforms=self.data_transforms['test'])



    def create_vis_healthy(self, transforms):
        idx = np.where(self.train_df[0:2000]["No Finding"] == 1)[0]
        filtered_df = self.train_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = CheXpert(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset

    def create_vis_anomaly(self, transforms):
        idx = np.where(self.train_df[0:2000]["No Finding"] != 1)[0]
        filtered_df = self.train_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = CheXpert(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset

    def vis_healthy_loader(self, batch_size, shuffle=False):
        return DataLoader(self.vis_healthy, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def vis_anomaly_loader(self, batch_size, shuffle=False):
        return DataLoader(self.vis_anomaly, batch_size=batch_size, shuffle=shuffle, drop_last=True)


    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, concat_testset=False, batch_size=1, shuffle=False):
        if concat_testset:
            self.concat_testset = ConcatDataset([self.test_set, self.official_vaild_set])
            return DataLoader(self.concat_testset, batch_size=batch_size, shuffle=shuffle)
        else:
            return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)

    def official_valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.official_vaild_set, batch_size=batch_size, shuffle=shuffle)



    def single_disease_train_dataloaders(self, batch_size, shuffle=True):
        train_dataloaders = {}
        for c in ['neg', 'pos']:
            train_loader = {}
            for disease in self.TRAIN_DISEASES:
                train_loader[disease] = DataLoader(self.single_disease_train_sets[c][disease], batch_size=batch_size, shuffle=shuffle, drop_last=True)
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


    def preprocess_df(self, file_name):
        file_path = os.path.join(self.image_dir, file_name)
        df = pd.read_csv(file_path)
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small/', '')
        # select orientation
        df = self.get_orientation(df, self.orientation)
        df = df.reset_index(drop=True)
        return df


    def fillnan_approach(self, df):
        new_df = df.fillna(0)
        return new_df

    def uncertainty_approach(self, df):
        # uncertainty labels are mapped to 0
        if self.uncertainty == 'toOne':
            new_df = df.replace(-1, 1)
        if self.uncertainty == 'toZero':
            new_df = df.replace(-1, 0)
        if self.uncertainty =='keep':
            new_df = df
        return new_df

    def create_trainsets(self):
        # create positive trainset and negative trainset for each disease
        train_sets = {}
        for c in ['neg', 'pos']:
            train_set_d = {}
            for disease in self.TRAIN_DISEASES:
                train_set_d[disease] = self.subset(src_df=self.train_df, disease=disease, label=c, transforms=self.data_transforms['train'])
            train_sets[c] = train_set_d
        return train_sets


    def create_vissets(self):
        # create positive and negative visualization set for each disease
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_df=self.train_df[0:1000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_df, disease, label, transforms):
        """
        Create positive or negative subset from source data frame for a given disease
        :param src_df: source data frame
        :param disease: str, the specific disease to filter
        :param label: str, 'neg' for negative samples, 'pos' for positive samples
        :param transforms: torchvision.transforms
        :return: a CheXpert Dataset object
        """

        if label == 'pos':
            idx = np.where(src_df[disease] == 1)[0]
        if label == 'neg':
            idx = np.where(src_df[disease] == 0)[0]
        filtered_df = src_df.iloc[idx]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = CheXpert(image_dir=self.image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms)
        return subset


    def get_orientation(self, df, orientation='Frontal'):
        # Deleting either lateral or frontal images of the Dataset or keep all
        if orientation == "Lateral":
            df = df[df['Frontal/Lateral'] == 'Lateral']
        elif orientation == "Frontal":
            df = df[df['Frontal/Lateral'] == 'Frontal']
        elif orientation == "all":
            df=df
        else:
            raise Exception("Wrong orientation input given!")
        return df



    def split(self, df, train_ratio, seed, shuffle=True):
        print("spliting data frame into trainset, validation set and test set....")
        path = df["Path"].tolist()
        patient_id = [p.split("/")[1] for p in path]
        unique_patient = np.unique(np.array(patient_id))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(unique_patient)

        split1 = int(np.floor(train_ratio * len(unique_patient)))
        split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(unique_patient)))
        train_patientID, valid_patientID, test_patientID = unique_patient[:split1], unique_patient[split1:split2], unique_patient[split2:]

        train_indices = []
        valid_indices = []
        test_indices = []
        for index, row in df.iterrows():
            patient_id = row["Path"].split("/")[1]
            if patient_id in train_patientID:
                train_indices.append(index)
            elif patient_id in valid_patientID:
                valid_indices.append(index)
            else:
                test_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        test_df = df.iloc[test_indices]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.split_df_dir, 'train_df.csv'), index=False)
        valid_df.to_csv(os.path.join(self.split_df_dir, 'valid_df.csv'), index=False)
        test_df.to_csv(os.path.join(self.split_df_dir, 'test_df.csv'), index=False)

        return train_df, valid_df, test_df






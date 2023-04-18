import os
import shutil
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from PIL import Image
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.data_utils import map_image_to_intensity_range, normalize_image

class Vindr_CXR(Dataset):
    def __init__(self, image_dir, df, train_diseases, transforms, img_size, with_BBox):
        self.image_dir = image_dir
        self.df = df
        self.TRAIN_DISEASES = train_diseases
        self.transforms = transforms
        self.img_size = img_size
        # create image list
        img_id = df['image_id'].tolist()
        self.image_list = np.unique(np.asarray(img_id))
        self.with_BBox = with_BBox
        if self.with_BBox:
            self.image_dir = "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train_pngs"


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        data = {}
        img_id = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_id + '.png')
        img = Image.open(img_path)  # value (0,255)

        image_size = img.size
        scale_factor_x = self.img_size / float(image_size[0])
        scale_factor_y = self.img_size / float(image_size[1])

        if self.transforms is not None:
            img = self.transforms(img) # value in range (0,1)

        img = normalize_image(img)
        img = map_image_to_intensity_range(img, -1, 1, percentiles=0.95)

        # Get labels from the dataframe for current image
        rows = self.df.loc[self.df['image_id'] == img_id]
        # lesion_types = rows['class_name'].tolist()

        if self.with_BBox:
            label = np.zeros(len(self.TRAIN_DISEASES))
            bbox = np.zeros((len(self.TRAIN_DISEASES), 4))
            for index, row in rows.iterrows():
                lesion_type = row['class_name']
                if lesion_type in self.TRAIN_DISEASES:
                    idx = self.TRAIN_DISEASES.index(lesion_type)
                    label[idx] = 1
                    bb = [int(row['x_min'] * scale_factor_x),
                          int(row['y_min'] * scale_factor_y),
                          int((row['x_max']-row['x_min'])* scale_factor_x),
                          int((row['y_max']-row['y_min'])* scale_factor_y)]

                    bbox[idx] = np.array(bb)
            data['img'] = img
            data['label'] = label
            data['BBox'] = bbox
        else:
            # Get labels from the dataframe for current image
            label = self.df.iloc[idx][self.TRAIN_DISEASES].values.tolist()
            label = np.array(label)
            # chexpert does not contain bounding box labels
            data['img'] = img
            data['label'] = label

        return data


class Vindr_CXRDataModule(LightningDataModule):

    def __init__(self, dataset_params, split_ratio=0.8, resplit=True, img_size=320, seed=42, with_bb=False):

        self.image_dir = dataset_params["image_dir"]
        self.train_csv_file = dataset_params["train_csv_file"]
        self.TRAIN_DISEASES = dataset_params["train_diseases"]
        self.train_image_dir = self.image_dir + "/train_pngs_rescaled"
        self.split_df_dir = os.path.join(self.image_dir, 'split_df')
        self.split_ratio = split_ratio
        self.resplit = resplit
        self.img_size = img_size
        self.seed = seed
        self.with_bb = with_bb
        self.diagnoses=[]

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
        # Read train and test csv files
        train_df = pd.read_csv(self.train_csv_file)
        self.diagnoses = np.unique(np.asarray(train_df['class_name'].tolist())).tolist()

        # Preprocess csv file
        # Split the train dataframe into train, valid and test dataframe
        if os.path.exists(self.split_df_dir) and self.resplit==False:
            print('Already split data, will use previous created split dataframe!')
            self.train_df = pd.read_csv(os.path.join(self.split_df_dir, 'train_df.csv'))
            self.valid_df = pd.read_csv(os.path.join(self.split_df_dir, 'valid_df.csv'))
            self.test_df = pd.read_csv(os.path.join(self.split_df_dir, 'test_df.csv'))

        else:
            if os.path.exists(self.split_df_dir):
                shutil.rmtree(self.split_df_dir)
            os.mkdir(self.split_df_dir)
            self.train_df, self.valid_df, self.test_df = self.split(df=train_df, train_ratio=self.split_ratio, shuffle=True)

        self.train_df, self.valid_df, self.test_df = self.restructure_cls_csv()
        self.BBox_test_df = self.test_df

        # Create datasets
        self.train_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.train_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['train'], img_size=self.img_size, with_BBox=False)
        self.valid_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.valid_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.test_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.test_df, train_diseases=self.TRAIN_DISEASES, transforms=self.data_transforms['test'], img_size=self.img_size, with_BBox=False)
        self.BBox_test_set = Vindr_CXR(image_dir=self.train_image_dir, df=self.BBox_test_df, train_diseases=self.TRAIN_DISEASES,
                                  transforms=self.data_transforms['test'], img_size=self.img_size,
                                  with_BBox=True)

        self.single_disease_train_sets = self.create_trainsets()
        self.single_disease_vis_sets = self.create_vissets()


    def BBox_test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.BBox_test_set, batch_size=batch_size, shuffle=shuffle)

    def train_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle)

    def valid_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.valid_set, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle)


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
        Create positive trainset and negative trainset for each disease
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
        Create positive trainset and negative visualization for each disease
        """
        vis_sets = {}
        for c in ['neg', 'pos']:
            vis_set_d = {}
            for disease in self.TRAIN_DISEASES:
                vis_set_d[disease] = self.subset(src_df=self.train_df[0:2000], disease=disease, label=c, transforms=self.data_transforms['test'])
            vis_sets[c] = vis_set_d
        return vis_sets


    def subset(self, src_df, disease, label, transforms):

        if self.with_bb:
            pos_idx = np.where(src_df['class_name'] == disease)[0]
        else:
            pos_idx = np.where(src_df[disease] == 1)[0]

        pos_img_list = src_df.iloc[pos_idx]['image_id'].tolist()
        unique_pos_img = np.unique(np.asarray(pos_img_list))

        if label == 'pos':
            filter_indices = []
            for index, row in src_df.iterrows():
                image_id = row['image_id']
                if image_id in unique_pos_img:
                    filter_indices.append(index)

        if label == 'neg':
            filter_indices = []
            for index, row in src_df.iterrows():
                image_id = row['image_id']
                if image_id in unique_pos_img:
                    pass
                else:
                    filter_indices.append(index)

        filtered_df = src_df.iloc[filter_indices]
        filtered_df = filtered_df.reset_index(drop=True)
        subset = Vindr_CXR(image_dir=self.train_image_dir, df=filtered_df, train_diseases=self.TRAIN_DISEASES, transforms=transforms, img_size=self.img_size, with_BBox=self.with_bb)

        return subset


    def split(self, df, train_ratio, shuffle=True):
        
        image_list = df['image_id'].tolist()
        unique_images = np.unique(np.asarray(image_list))
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(unique_images)

        split1 = int(np.floor(train_ratio * len(unique_images)))
        split2 = split1 + int(np.floor(0.5 * (1 - train_ratio) * len(unique_images)))
        train_images, valid_images, test_images = unique_images[:split1], unique_images[split1:split2], unique_images[split2:]

        train_indices = []
        valid_indices = []
        test_indices = []
        for index, row in df.iterrows():
            img_id = row['image_id']
            if img_id in train_images:
                train_indices.append(index)
            elif img_id in valid_images:
                valid_indices.append(index)
            else:
                test_indices.append(index)

        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        test_df = df.iloc[test_indices]

        # reset index to get continuous index from 0
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.split_df_dir, "train_df.csv"))
        valid_df.to_csv(os.path.join(self.split_df_dir, "valid_df.csv"))
        test_df.to_csv(os.path.join(self.split_df_dir, "test_df.csv"))

        return train_df, valid_df, test_df


    def restructure_cls_csv(self):
        # Rewrite the original csv files. columns are 28 labels + image_id, each row is a image.
        # Restructuring the csv files is to make it easier to create dataloader for classification task.
        train_df = self.create_df(self.train_df, 'train_cls_labels.csv')
        valid_df = self.create_df(self.valid_df, 'valid_cls_labels.csv')
        test_df = self.create_df(self.test_df, 'test_cls_labels.csv')

        return train_df, valid_df, test_df


    def create_df(self, df , filename):
        unique_img_id = np.unique(df['image_id'].tolist())
        columns = self.diagnoses
        new_df = pd.DataFrame(0.0, index=np.arange(len(unique_img_id)), columns=columns)
        new_df['image_id'] = unique_img_id

        for i, row in new_df.iterrows():
            img_id = row['image_id']
            rows = df.loc[df['image_id'] == img_id]
            for j, r in rows.iterrows():
                lesion_type = r['class_name']
                row[lesion_type] = 1.0
            new_df.loc[i] = row

        path = os.path.join(self.split_df_dir, filename)
        new_df.to_csv(path)
        return new_df









chexpert_dict = {
"image_dir": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/",
"train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/train.csv",
"test_csv_file": "/mnt/qb/work/baumgartner/sun22/data/CheXpert-v1.0-small/valid.csv",
"train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
"orientation": "Frontal",
"uncertainty": "toZero"
}

nih_chestxray_dict = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled",
    "data_entry_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/Data_Entry_2017.csv",
    "BBox_csv_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/BBox_List_2017.csv",
    "train_valid_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/train_val_list.txt",
    "test_list_file": "/mnt/qb/work/baumgartner/sun22/data/NIH_labels/test_list.txt",
    "train_diseases": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

}

vindr_cxr = {
    "image_dir": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection",
    "train_csv_file": "/mnt/qb/work/baumgartner/sun22/data/Vindr-CXR/vinbigdata-chest-xray-abnormalities-detection/train.csv",
    "train_diseases": ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion'],
}

data_default_params = {
    "split_ratio": 0.8,
    "resplit": False,
    "img_size": 320,
}

dataset_dict = {
    "chexpert": chexpert_dict,
    "nih_chestxray": nih_chestxray_dict,
    "vindr_cxr": vindr_cxr
}












import os
import numpy as np
import torch
import random
import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
from train_utils import to_numpy
from tqdm import tqdm


def str2bool(v):
    return v.lower() in ('true')


class class_sensitivity_analyser():
    def __init__(self, solver, config):
        self.solver = solver
        self.result_dir = config.result_dir
        self.train_diseases = self.solver.TRAIN_DISEASES
        self.best_threshold = {}
        self.attr_method = config.attr_method
        os.makedirs(self.result_dir, exist_ok=True)
        self.attr_dir = os.path.join(self.result_dir, self.attr_method)
        os.makedirs(self.attr_dir, exist_ok=True)

        print('compute class sensitivity on dataset ' + config.dataset)
        print('use explainer: ' + self.attr_method)

        # Read data from file.
        threshold_path = os.path.join(self.result_dir,"best_threshold.txt")
        if os.path.exists(threshold_path):
            print("threshold alreay computed, load threshold")
            threshold = np.loadtxt(open(threshold_path))
            for c in range(len(self.solver.TRAIN_DISEASES)):
                disease = self.solver.TRAIN_DISEASES[c]
                self.best_threshold[disease] = threshold[c]

        else:
            self.best_threshold = solver.get_optimal_thresholds(save_result=True, result_dir=self.result_dir)
        pred_file = os.path.join(self.result_dir,"test_pred.txt")

        if os.path.exists(pred_file):
            print("prediction of test set already made")
            self.test_pred = np.loadtxt(os.path.join(self.result_dir,"test_pred.txt"))
            self.test_true = np.loadtxt(os.path.join(self.result_dir,"test_true.txt"))
        else:
            self.test_pred, self.test_true, class_auc = self.solver.test(which_loader="test", save_result=True, result_dir=self.result_dir)
            print('test_class_auc', class_auc)

    def filter_correct_pred(self, test_pred, test_true, train_disease, best_threshold):
        pred = np.zeros(test_pred.shape)
        for i in range(len(train_disease)):
            disease = train_disease[i]
            pred[np.where(test_pred[:, i] > best_threshold[disease])[0], i] = 1
            pred[np.where(test_pred[:, i] < best_threshold[disease])[0], i] = 0
        results = {}
        for i in range(len(train_disease)):
            r = {}
            disease = train_disease[i]
            pos_idx = np.where(pred[:, i] == 1)[0]
            neg_idx = np.where(pred[:, i] == 0)[0]
            t_pos_idx = np.where(test_true[:, i] == 1)[0]
            t_neg_idx = np.where(test_true[:, i] == 0)[0]
            TP = list(set(pos_idx.tolist()).intersection(t_pos_idx.tolist()))
            TN = list(set(neg_idx.tolist()).intersection(t_neg_idx.tolist()))
            tp_pred = np.column_stack((TP, test_pred[:,i][TP]))
            tn_pred = np.column_stack((TN, test_pred[:,i][TN]))
            sorted_tp_pred = tp_pred[tp_pred[:, 1].argsort()] # from smallest to biggest
            sorted_tn_pred = tn_pred[tn_pred[:, 1].argsort()] # from smallest to biggest
            sorted_TP = sorted_tp_pred[:,0]
            sorted_TN = sorted_tn_pred[:,0]
            r['TP'] = sorted_TP[::-1] # from biggest to smallest
            r['TN'] = sorted_TN # from smallest to biggest
            results[disease] = r
        return results

    def create_grids(self, n_cells, pred_dict, train_disease, num_imgs):
        # n_cells is 2 or 3. to make grid 2*2 or 3*3
        grids = {}
        num_neg = n_cells * n_cells - 1
        for disease in train_disease:
            blocks = []
            preds = pred_dict[disease]
            TP = preds['TP'][:num_imgs].tolist()
            TN = preds['TN'][:num_imgs*num_neg].tolist()

            random.shuffle(TN)
            current_point = 0
            for idx in TP:
                b = []
                b.append(idx)
                if (current_point+num_neg) <= len(TN):
                    for j in range(current_point, current_point+num_neg):
                        b.append(TN[j])
                    current_point += num_neg
                else:
                    current_point=0
                blocks.append(b)
            grids[disease] = blocks
        return grids

    def compute_localization_score(self, idx_grids, attr_methods):
        scores = {}
        mean = 0
        for disease in idx_grids.keys():
            score_list = []
            blocks = idx_grids[disease]
            for i in tqdm(range(len(blocks))):
                b = blocks[i]
                sc = self.compute_sc(b, disease, attr_methods)
                score_list.append(sc)
            avg_score = np.mean(np.array(score_list))
            scores[disease] = avg_score
            mean += avg_score

        mean = mean / (len(idx_grids.keys()))
        print('mean localization score on all disease: ', mean)

        return scores


    def compute_sc(self, index_list, disease, attr_methods):
        label_idx = self.solver.TRAIN_DISEASES.index(disease)
        pixel_counts = []
        for idx in index_list:
            data = self.solver.test_loader.dataset[int(idx)]
            img = data['img']
            img = torch.from_numpy(img[None])
            attr = self.solver.get_attributes(img, label_idx)
            attr = to_numpy(attr).squeeze()
            if attr_methods in ['attri-net']:
                sum_pixel = np.sum(abs(attr))
            if attr_methods in ['lime', 'GB', 'GCam', 'shap', 'gifsplanation']:
                sum_pixel = np.sum(attr)
            pixel_counts.append(sum_pixel)

        if np.sum(np.array(pixel_counts)) !=0 :
            score = pixel_counts[0] / (np.sum(np.array(pixel_counts)))
        else:
            score = 0
        return score


    def run(self):
        # Filter truly predicted images
        filter_results = self.filter_correct_pred(self.test_pred, self.test_true, train_disease=self.train_diseases, best_threshold=self.best_threshold)
        # Creating 2*2 blocks
        idx_grids = self.create_grids(2, filter_results, self.train_diseases, num_imgs=200)
        # Compute class sensitivity score.
        local_score = self.compute_localization_score(idx_grids, attr_methods=self.attr_method)
        print(local_score)









def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='resnet', choices=['resnet', 'attri-net'])
    parser.add_argument('--attr_method', type=str, default='shap',
                        help="choose the explaination methods, can be 'lime', 'GCam', 'GB', 'shap', 'attri-net' , 'gifsplanation'")

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='vindr_cxr', choices=['chexpert', 'nih_chestxray', 'vindr_cxr'])
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()

    # Change the path to your own path.
    resnet_model_path_dict = {
        "chexpert": "/path/to/your/resnet_model/on/chexpert",
        "nih_chestxray": "/path/to/your/resnet_model/on/nih_chestxray",
        "vindr_cxr": "/path/to/your/resnet_model/on/vindr_cxr"
    }

    attrinet_model_path_dict = {
        "chexpert": "/path/to/your/attrinet_model/on/chexpert",
        "nih_chestxray": "/path/to/your/attrinet_model/on/nih_chestxray",
        "vindr_cxr": "/path/to/your/attrinet_model/on/vindr_cxr"
    }

    if opts.exp_name == 'resnet':
        opts.model_path = resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)
        opts.result_dir = os.path.join(opts.model_path, "result_dir")

    if opts.exp_name == 'attri-net':
        opts.model_path = attrinet_model_path_dict[opts.dataset]
        print("evaluate attri-net model: " + opts.model_path)
        opts.result_dir = os.path.join(opts.model_path, "result_dir")
        # Configurations of networks
        opts.image_size = 320
        opts.n_fc = 8
        opts.n_ones = 20
        opts.num_out_channels = 1
        opts.lgs_downsample_ratio = 32

    return opts


def prep_solver(datamodule, exp_configs):
    data_loader = {}
    if "resnet" in exp_configs.exp_name:
        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        solver = resnet_solver(exp_configs, data_loader=data_loader)
        solver.set_explainer(which_explainer=exp_configs.attr_method)

    if "attri-net" in exp_configs.exp_name:
        train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
        vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=exp_configs.batch_size, shuffle=False)
        valid_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

        data_loader['train_pos'] = train_loaders['pos']
        data_loader['train_neg'] = train_loaders['neg']
        data_loader['vis_pos'] = vis_dataloaders['pos']
        data_loader['vis_neg'] = vis_dataloaders['neg']
        data_loader['valid'] = valid_loader
        data_loader['test'] = test_loader
        solver = task_switch_solver(exp_configs, data_loader=data_loader)
    return solver


def main(config):
    from data.dataset_params import dataset_dict, data_default_params
    datamodule = prepare_datamodule(config, dataset_dict, data_default_params)
    solver = prep_solver(datamodule, config)
    analyser = class_sensitivity_analyser(solver, config)
    analyser.run()


if __name__ == "__main__":
    params = get_arguments()
    main(params)
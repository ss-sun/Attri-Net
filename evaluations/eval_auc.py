import sys
import os
sys.path.append(os.path.abspath("/mnt/qb/work/baumgartner/sun22/project/MT_VAGAN"))

import argparse
from train_utils import prepare_datamodule
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver

def str2bool(v):
    return v.lower() in ('true')

class classification_analyser():
    """
    Analyse the classification performance of the model.
    """
    def __init__(self, solver):
        self.solver = solver

    def run(self):
        _, _, auc = self.solver.test()
        return auc

def argument_parser():
    """
    Create a parser with experiments arguments.
    """

    parser = argparse.ArgumentParser(description="classification metric analyser.")
    parser.add_argument('--exp_name', type=str, default='resnet', choices=['resnet', 'attri-net'])
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

    # change the path to your own path.
    # resnet_model_path_dict = {
    #     "chexpert": "/path/to/your/resnet_model/on/chexpert",
    #     "nih_chestxray": "/path/to/your/resnet_model/on/nih_chestxray",
    #     "vindr_cxr": "/path/to/your/resnet_model/on/vindr_cxr"
    # }

    # attrinet_model_path_dict = {
    #     "chexpert": "/path/to/your/attrinet_model/on/chexpert",
    #     "nih_chestxray": "/path/to/your/attrinet_model/on/nih_chestxray",
    #     "vindr_cxr": "/path/to/your/attrinet_model/on/vindr_cxr"
    # }

    resnet_model_path_dict = {
        "chexpert": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-17 10:40:08-chexpert-bs=8-lr=0.0001-weight_decay=1e-05",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:22-nih_chestxray-bs=8-lr=0.0001-weight_decay=1e-05",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/resnet/resnet2023-04-13 18:16:34-vindr_cxr-bs=8-lr=0.0001-weight_decay=1e-05"
    }

    attrinet_model_path_dict = {
        "chexpert": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 17:52:55--chexpert--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "nih_chestxray": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-14 10:57:38--nih_chestxray--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01",
        "vindr_cxr": "/mnt/qb/work/baumgartner/sun22/repo_exps/attri-net/attri-net2023-04-13 18:10:38--vindr_cxr--bs=4--lg_ds=32--l_cri=1.0--l1=100--l2=200--l3=100--l_ctr=0.01"
    }


    if opts.exp_name == 'resnet':
        opts.model_path = resnet_model_path_dict[opts.dataset]
        print("evaluate resnet model: " + opts.model_path)

    if opts.exp_name == 'attri-net':
        opts.model_path = attrinet_model_path_dict[opts.dataset]
        print("evaluate attri-net model: " + opts.model_path)

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
    analyser = classification_analyser(solver)
    analyser.run()


if __name__ == "__main__":

    params = get_arguments()
    main(params)
from solvers.resnet_solver import resnet_solver
from solvers.attrinet_solver import task_switch_solver
import logging
from experiment_utils import init_seed, init_experiment, init_wandb
from train_utils import prepare_datamodule



def prepare_exps(exp_configs):
    if exp_configs.mode == 'train':
        print("training model: ")
        init_experiment(exp_configs)
    init_seed(exp_configs.manual_seed)
    if exp_configs.use_wandb:
        init_wandb(exp_configs)



def main(exp_configs):
    from data.dataset_params import dataset_dict, data_default_params
    prepare_exps(exp_configs)
    print("experiment folder: " + exp_configs.exp_dir)
    datamodule = prepare_datamodule(exp_configs, dataset_dict, data_default_params)
    print(exp_configs)

    # Prepare data loaders and solver.
    data_loader = {}
    if "resnet" in exp_configs.exp_name:
        print("working on resnet")
        data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
        data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        solver = resnet_solver(exp_configs, data_loader=data_loader)

    if "attri-net" in exp_configs.exp_name:
        print("working on attri-net")
        train_loaders = datamodule.single_disease_train_dataloaders(batch_size=exp_configs.batch_size, shuffle=True)
        vis_dataloaders = datamodule.single_disease_vis_dataloaders(batch_size=4, shuffle=False)
        valid_loader = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
        test_loader = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)

        data_loader['train_pos'] = train_loaders['pos']
        data_loader['train_neg'] = train_loaders['neg']
        data_loader['vis_pos'] = vis_dataloaders['pos']
        data_loader['vis_neg'] = vis_dataloaders['neg']
        data_loader['valid'] = valid_loader
        data_loader['test'] = test_loader
        solver = task_switch_solver(exp_configs, data_loader=data_loader)

    if exp_configs.mode == "train":
        print('start training...')
        solver.train()
        print('finish training!')

    if exp_configs.mode == 'test':
        print('start testing....')
        solver.load_model(exp_configs.test_model_path)
        test_auc = solver.test()
        print('finish test!')
        print('test_auc: ', test_auc)





if __name__ == '__main__':
    from parser import resnet_get_parser, attrinet_get_parser
    model = "resnet" # select which model to train, "resnet" or "attri-net"
    if model == "resnet":
        parser = resnet_get_parser()
    if model == "attri-net":
        parser = attrinet_get_parser()
    config = parser.parse_args()
    main(config)
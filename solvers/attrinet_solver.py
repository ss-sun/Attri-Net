import os
import torch
from torch import autograd
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from train_utils import to_numpy, save_batch, logscalar
from models.attrinet_modules import Discriminator_with_Ada, Generator_with_Ada, CenterLoss
from models.lgs_classifier import LogisticRegressionModel
import matplotlib.pyplot as plt
from tqdm import tqdm




class task_switch_solver(object):
    """
    Train and test task switching Attri-Net.
    Assign task code to net_g and net_d for task-specific functionality.
    When a task is switched on, net_g generates a mask relevant to that specific task, while the discriminator produces a corresponding value for that mask.
    All hyperparameters are provided by the user in the config file.
    """

    def __init__(self, exp_configs, data_loader):
        """
        Initialize solver
        """

        # Initialize configurations.
        self.exp_configs = exp_configs
        self.use_gpu = exp_configs.use_gpu
        self.mode = exp_configs.mode

        # Set device.
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Model configurations.
        self.TRAIN_DISEASES = exp_configs.train_diseases
        self.dataloaders = data_loader
        self.current_training_disease = None

        # Configurations of generator.
        self.img_size = exp_configs.image_size
        self.num_class = len(self.TRAIN_DISEASES)

        # Configurations of latent code generator.
        self.n_fc = exp_configs.n_fc
        self.n_ones = exp_configs.n_ones
        self.num_out_channels = exp_configs.num_out_channels

        # Configurations of classifiers.
        self.logreg_dsratio = exp_configs.lgs_downsample_ratio
        if self.mode == "train":

            # Training configurations.
            self.lambda_critic = exp_configs.lambda_critic
            self.lambda_1 = exp_configs.lambda_1
            self.lambda_2 = exp_configs.lambda_2
            self.lambda_3 = exp_configs.lambda_3
            self.lambda_ctr = exp_configs.lambda_centerloss
            self.d_iters = exp_configs.d_iters # more discriminator steps for one generator step
            self.cls_iteration = exp_configs.cls_iteration # more classifier steps for one generator step
            self.num_iters = exp_configs.num_iters # total generator steps
            self.batch_size = exp_configs.batch_size
            self.g_lr = exp_configs.g_lr
            self.d_lr = exp_configs.d_lr
            self.lgs_lr = exp_configs.lgs_lr
            self.weight_decay_lgs = exp_configs.weight_decay_lgs
            self.beta1 = exp_configs.beta1
            self.beta2 = exp_configs.beta2

            # Step size.
            self.sample_step = exp_configs.sample_step
            self.model_valid_step = exp_configs.model_valid_step
            self.lr_update_step = exp_configs.lr_update_step

            # Directories.
            self.ckpt_dir = exp_configs.ckpt_dir
            self.output_dir = exp_configs.output_dir

            # Data params
            self.dloader_pos = data_loader['train_pos']
            self.dloader_neg = data_loader['train_neg']
            self.valid_loader = data_loader['valid']
            self.vis_loader_pos = data_loader['vis_pos']
            self.vis_loader_neg = data_loader['vis_neg']

            # Create latent code for self.TRAIN_DISEASES tasks
            self.latent_z_task = self.create_task_code()

            # Initialize the models.
            self.init_model(self.device)
            self.net_g.apply(self.weights_init)
            self.net_d.apply(self.weights_init)
            self.optim_g, self.optim_d, self.optim_lgs = self.init_optimizer(self.net_g, self.net_d, self.net_lgs)
            self.lgs_loss = torch.nn.BCEWithLogitsLoss()
            self.center_losses, self.optimizer_centloss = self.init_centerlosses()

            # Miscellaneous.
            self.use_wandb = exp_configs.use_wandb

        if self.mode == "test":
            self.vis_loader_pos = data_loader['vis_pos']
            self.vis_loader_neg = data_loader['vis_neg']
            self.valid_loader = data_loader['valid']
            self.test_loader = data_loader['test']
            self.model_path = exp_configs.model_path

            # Create latent code for self.TRAIN_DISEASES tasks
            self.latent_z_task = self.create_task_code()

            # Load the trained models.
            self.init_model(self.device)
            self.load_model(self.model_path, is_best=True)



    def init_centerlosses(self):
        """
        Initialize centerlosses for each task.
        """
        center_losses = {}
        optimizer_centloss = {}
        for disease in self.TRAIN_DISEASES:
            loss = CenterLoss(num_classes=2, feat_dim=self.img_size*self.img_size, device=self.device)
            opt = torch.optim.SGD(loss.parameters(), lr=0.1)
            center_losses[disease] = loss
            optimizer_centloss[disease] = opt
        return center_losses, optimizer_centloss


    def create_task_code(self):
        """
        Create code for each task, e.g. for task 0, the code is [1, 1, ..., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        latent_z_task = {}
        for i, task in enumerate(self.TRAIN_DISEASES):
            z = torch.zeros([1, self.num_class * self.n_ones], dtype=torch.float)
            start = self.n_ones * i
            end = self.n_ones * (i+1)
            z[:, start:end] = 1
            latent_z_task[task] = z
        return latent_z_task


    def weights_init(self, m):
        """
        Initialize cnn weights.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data, 2)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def init_model(self, device):
        """
        Initialize generator, disciminator and classifiers.
        """
        print('Initialize networks.')
        self.net_g = Generator_with_Ada(num_classes=self.num_class, img_size=self.img_size, num_masks=1,
                                        act_func="relu", n_fc=8, dim_latent=self.num_class * self.n_ones,
                                        conv_dim=64, in_channels=1, repeat_num=6)
        self.net_d = Discriminator_with_Ada(act_func="relu", conv_dim=64, dim_latent=self.num_class * self.n_ones,
                                       repeat_num=6)

        self.net_g.to(device)
        self.net_d.to(device)

        # Initialize one classifier for each disease.
        self.net_lgs = {}
        for disease in self.TRAIN_DISEASES:
            m = LogisticRegressionModel(
                input_size=self.img_size, num_classes=1, downsample_ratio=self.logreg_dsratio)
            m.to(device)
            self.net_lgs[disease] = m


    def init_optimizer(self, net_g, net_d, net_lgs):
        """
        Initialize optimizers
        """
        optimizer_g = optim.Adam(
            net_g.parameters(),
            lr=self.g_lr,
            betas=(self.beta1, 0.9),
            weight_decay=1e-5,
        )
        optimizer_d = optim.Adam(
            net_d.parameters(),
            lr=self.d_lr,
            betas=(self.beta1, 0.9),
            weight_decay=1e-5,
        )
        optimizer_lgs = {}
        for disease in self.TRAIN_DISEASES:
            opt = torch.optim.Adam(
                    net_lgs[disease].parameters(),
                    lr=self.lgs_lr,
                    weight_decay=self.weight_decay_lgs,
            )
            optimizer_lgs[disease] = opt
        return optimizer_g, optimizer_d, optimizer_lgs

    def calc_gradient_penalty(self, netD, task_code, real_data, fake_data):
        """
        Calculate gradient penalty as in  "Improved Training of Wasserstein GANs"
        https://github.com/caogang/wgan-gp
        """
        LAMBDA = 10
        bs, ch, h, w = real_data.shape
        use_cuda = real_data.is_cuda
        alpha = torch.rand(bs, 1)
        alpha = (
            alpha.expand(bs, int(real_data.nelement() / bs)).contiguous().view(bs, ch, h, w)
        )
        alpha = alpha.to(self.device)
        interpolates = torch.tensor(
            alpha * real_data + ((1 - alpha) * fake_data), requires_grad=True
        )
        interpolates = interpolates.to(self.device)
        disc_interpolates = netD(interpolates, task_code)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device)
            if use_cuda
            else torch.ones(disc_interpolates.size()),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty


    def set_require_grads(self, grad_flags):
        """
        Set requires_grad of generator, discriminator, and classifiers to only update one of them.
        """
        # Set requires_grad of net_d to True only during net_d update and to False during net_g or net_lgs update.
        for p in self.net_d.parameters():
            p.requires_grad = grad_flags[0]

        # Set requires_grad of net_g to to True only during net_g update False during net_d or net_lgs update.
        for p in self.net_g.parameters():
            p.requires_grad = grad_flags[1]

        # Set requires_grad of the currently training classifier to True and other classifiers require grad to False during net_lgs update.
        for disease in self.TRAIN_DISEASES:
            model = self.net_lgs[disease]
            if disease == self.current_training_disease:
                for p in model.parameters():
                    p.requires_grad = grad_flags[2]
            else:
                for p in model.parameters():
                    p.requires_grad = False



    def save_checkpoint(self, step, is_best):
        """
        Save the latest model or the best model.
        """
        if is_best:
            suffix = '_best.pth'
            print("save best models at generator iteration: ", step)
        else:
            suffix = '_last.pth'
            print("save last models at generator iteration: ", step)

        torch.save(
            self.net_g.state_dict(),
            f"{self.ckpt_dir}/net_g"+suffix,
        )
        torch.save(
            self.net_d.state_dict(),
            f"{self.ckpt_dir}/net_d"+suffix,
        )
        for disease in self.TRAIN_DISEASES:
            torch.save(
                self.net_lgs[disease].state_dict(),
                f"{self.ckpt_dir}/classifier_of_{disease}"+suffix,
                )
            torch.save(
                self.center_losses[disease].state_dict(),
                f"{self.ckpt_dir}/center_of_{disease}" + suffix,
            )



    def load_model(self, model_path, is_best=True):
        """
        Restore the trained generator, discriminator, classifiers and class centers.
        """
        model_dir = model_path + "/ckpt"
        if is_best:
            print('loading the best model...')
            suffix = '_best.pth'
        else:
            print('loading the best model...')
            suffix = '_last.pth'

        net_g_path = os.path.join(model_dir, 'net_g'+suffix)
        self.net_g.load_state_dict(torch.load(net_g_path))

        net_d_path = os.path.join(model_dir, 'net_d'+suffix)
        self.net_d.load_state_dict(torch.load(net_d_path))

        for c in range(self.num_class):
            disease = self.TRAIN_DISEASES[c]
            c_file_name = "classifier_of_" + disease +suffix
            c_path = os.path.join(model_dir, c_file_name)
            self.net_lgs[disease].load_state_dict(torch.load(c_path))
            # center_file_name = "center_of_" + disease + suffix
            # center_path = os.path.join(model_dir, center_file_name)
            # self.center_losses[disease].load_state_dict(torch.load(center_path))



    def save_vis_samples(self):
        """
        Save the input samples for visualization.
        """
        for disease in self.TRAIN_DISEASES:
            pos_iter = iter(self.vis_loader_pos[disease])
            neg_iter = iter(self.vis_loader_neg[disease])
            pos_data = next(pos_iter)
            pos_imgs, pos_lbls = pos_data['img'], pos_data['label']
            neg_data = next(neg_iter)
            neg_imgs, neg_lbls = neg_data['img'], neg_data['label']
            samples = torch.cat((pos_imgs, neg_imgs))
            lbls = torch.cat((pos_lbls, neg_lbls))
            samples_name = disease + "_input_samples.png"
            path = os.path.join(self.output_dir, samples_name)
            save_batch(to_numpy(samples * 0.5 + 0.5), to_numpy(lbls), None, path) # change values in samples to range [0,1] for visualization


    def save_disease_masks(self, gen_iterations):
        """
        Visualize the disease masks and the dest images during training.
        """
        for disease in self.TRAIN_DISEASES:
            classifier = self.net_lgs[disease]
            pos_iter = iter(self.vis_loader_pos[disease])
            neg_iter = iter(self.vis_loader_neg[disease])
            pos_data = next(pos_iter)
            pos_imgs, pos_lbls = pos_data['img'], pos_data['label']
            neg_data = next(neg_iter)
            neg_imgs, neg_lbls = neg_data['img'], neg_data['label']
            pos_imgs = pos_imgs.to(self.device)
            neg_imgs = neg_imgs.to(self.device)
            task_code = self.latent_z_task[disease]
            task_code = task_code.to(self.device)
            dests_pos, masks_pos = self.net_g(pos_imgs, task_code)
            dests_neg, masks_neg = self.net_g(neg_imgs, task_code)
            masks = torch.cat((masks_pos, masks_neg))
            dests = torch.cat((dests_pos, dests_neg))
            lbls = torch.cat((pos_lbls, neg_lbls))
            y_pred = classifier(masks)
            pred_batch = torch.sigmoid(y_pred)
            dest_name = str(gen_iterations) + "_" + disease + "_dest_samples.png"
            mask_name = str(gen_iterations) + "_" + disease + "_mask_samples.png"
            dest_path = os.path.join(self.output_dir, dest_name)
            mask_path = os.path.join(self.output_dir, mask_name)
            save_batch(to_numpy(dests * 0.5 + 0.5), to_numpy(lbls), to_numpy(pred_batch), dest_path)
            save_batch(to_numpy(-masks), to_numpy(lbls), to_numpy(pred_batch), mask_path)

    def vis_classcenter(self, gen_iterations):
        """
        Visualize the class centers during training.
        """
        for disease in self.TRAIN_DISEASES:
            loss_module = self.center_losses[disease]
            neg_center = loss_module.centers[0].data
            pos_center = loss_module.centers[1].data
            neg_center = to_numpy(neg_center).reshape((self.img_size, self.img_size))
            pos_center = to_numpy(pos_center).reshape((self.img_size, self.img_size))
            filename = str(gen_iterations) + "_" + disease + "_centers.png"
            out_dir = os.path.join(self.ckpt_dir, filename)
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4 * 1))
            axs[0].imshow(neg_center, cmap="gray")
            title = "neg center of " + disease
            axs[0].set_title(title)
            axs[0].axis('off')
            axs[1].imshow(pos_center, cmap="gray")
            title = "pos center of " + disease
            axs[1].set_title(title)
            axs[1].axis('off')
            plt.savefig(out_dir, bbox_inches='tight')


    def set_mode(self, mode):
        """
        Set the 'train' or 'eval' mode of the networks.
        """
        if mode == "train":
            self.net_g.train()
            self.net_d.train()
            for disease in self.TRAIN_DISEASES:
                self.net_lgs[disease].train()
        if mode == 'eval':
            self.net_g.eval()
            self.net_d.eval()
            for disease in self.TRAIN_DISEASES:
                self.net_lgs[disease].eval()


    def get_batch(self, current_training_disease, which_batch):
        """

        :param current_training_disease:
        :param which_batch:
        :return:
        """
        if which_batch == "pos":
            try:
                batch = next(self.pos_disease_data_iters[current_training_disease])
            except StopIteration:
                self.pos_disease_data_iters[current_training_disease] = iter(
                    self.dloader_pos[current_training_disease])
                batch = next(self.pos_disease_data_iters[current_training_disease])
        elif which_batch == "neg":
            try:
                batch = next(self.neg_disease_data_iters[current_training_disease])
            except StopIteration:
                self.neg_disease_data_iters[current_training_disease] = iter(
                    self.dloader_neg[current_training_disease])
                batch = next(self.neg_disease_data_iters[current_training_disease])

        imgs, lbls = batch['img'], batch['label']
        imgs = imgs.to(self.device)
        lbls = lbls.to(self.device)

        return imgs, lbls



    def train(self):
        # Fetch fixed inputs for visualization.
        self.save_vis_samples()

        # Prepare dataloaders.
        self.neg_disease_data_iters = {}
        self.pos_disease_data_iters = {}
        for disease in self.TRAIN_DISEASES:
            self.neg_disease_data_iters[disease] = iter(self.dloader_neg[disease])
            self.pos_disease_data_iters[disease] = iter(self.dloader_pos[disease])

        # Record classification performance(auc) during training.
        valid_auc = []
        best_auc = 0

        # Use number of iterations to control the training of networks.
        dis_iterations = 0
        classifier_iterations = 0

        # Start training.
        for gen_iterations in range(0, self.num_iters):

            # Train the discriminator d_iters times for one generator iteration.
            if gen_iterations < 25 or gen_iterations % 100 == 0:
                d_iters = 100
                print(" - Doing critic update steps ({} steps)".format(d_iters))
            else:
                d_iters = self.d_iters

            # =================================================================================== #
            #                             1. Train the discriminator                              #
            # =================================================================================== #

            # Only allow the discriminator to be trained.
            self.set_require_grads([True, False, False])

            for _ in range(d_iters):

                # Select a disease to train on and get a positive batch and a negative batch.
                self.current_training_disease = self.TRAIN_DISEASES[dis_iterations % self.num_class]
                imgs_neg, lbls_neg = self.get_batch(self.current_training_disease, which_batch="neg")
                imgs_pos, lbls_pos = self.get_batch(self.current_training_disease, which_batch="pos")

                # Switch the discriminator to a specific disease task and compute critic loss.
                task_code = self.latent_z_task[self.current_training_disease].to(self.device)
                dest, mask = self.net_g(imgs_pos, task_code)
                cri_loss = - self.net_d(imgs_neg, task_code).mean()
                cri_loss += self.net_d(dest, task_code).mean()
                cri_loss += self.calc_gradient_penalty(self.net_d, task_code, imgs_pos, dest.data)

                # Critic update step.
                self.optim_d.zero_grad()
                cri_loss.backward()
                self.optim_d.step()
                dis_iterations += 1
                print("dis_iterations", dis_iterations)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            # Only allow the generator to be trained.
            self.set_require_grads([False, True, False])

            # Select a disease to train on and get a positive batch and a negative batch.
            self.current_training_disease = self.TRAIN_DISEASES[gen_iterations % self.num_class]
            imgs_neg, lbls_neg = self.get_batch(self.current_training_disease, which_batch="neg")
            imgs_pos, lbls_pos = self.get_batch(self.current_training_disease, which_batch="pos")

            # Switch the generator to a specific disease task and compute generator loss.
            task_code= self.latent_z_task[self.current_training_disease].to(self.device)
            neg_dests, neg_masks = self.net_g(imgs_neg, task_code)
            pos_dests, pos_masks = self.net_g(imgs_pos, task_code)

            # Discriminator loss.
            gen_loss_d = - self.net_d(pos_dests, task_code).mean() * self.lambda_critic

            # Regularization terms.
            l1_anomaly = torch.mean(torch.abs(torch.mean(pos_masks, dim=1, keepdim=True))) * self.lambda_1
            l1_health = torch.mean(torch.abs(torch.mean(neg_masks, dim=1, keepdim=True))) * self.lambda_2

            # Classifier loss.
            masks_all = torch.cat((neg_masks, pos_masks))
            lbls_all = torch.cat((lbls_neg, lbls_pos))
            disease_idx = self.TRAIN_DISEASES.index(self.current_training_disease)
            classifier = self.net_lgs[self.current_training_disease]
            pred_all = classifier(masks_all)
            classifiers_loss = self.lgs_loss(pred_all.squeeze(), lbls_all[:, disease_idx])
            classifiers_loss = classifiers_loss * self.lambda_3

            # Get center loss.
            center_loss = self.center_losses[self.current_training_disease](masks_all.view(masks_all.size(0), -1),
                                                                            lbls_all[:, disease_idx]) * self.lambda_ctr
            # Total loss.
            gen_loss = gen_loss_d + l1_anomaly + l1_health + classifiers_loss + center_loss

            logdict = {
                "gen_loss": gen_loss,
                "gen_loss_d": gen_loss_d,
                "l1_anomaly": l1_anomaly,
                "l1_health": l1_health,
                "classifiers_loss": classifiers_loss,
                "center_loss": center_loss
            }
            for n, v in logdict.items():
                logscalar(f"per_batch.{n}", v)

            # Update generator.
            self.optim_g.zero_grad()
            if self.lambda_ctr != 0.0:
                self.optimizer_centloss[self.current_training_disease].zero_grad()
            gen_loss.backward()
            self.optim_g.step()
            if self.lambda_ctr != 0.0:
                for param in self.center_losses[self.current_training_disease].parameters():
                    param.grad.data *= (1. / self.lambda_ctr)
                self.optimizer_centloss[self.current_training_disease].step()

            gen_iterations += 1
            print("gen_iterations", gen_iterations)

            # =================================================================================== #
            #                               3. Train the classifiers                              #
            # =================================================================================== #

            for cls_step in range(self.cls_iteration):
                self.current_training_disease = self.TRAIN_DISEASES[classifier_iterations % self.num_class]
                # Set requires_grad for respective classifier.
                self.set_require_grads([False, False, True])

                imgs_neg, lbls_neg = self.get_batch(self.current_training_disease, which_batch="neg")
                imgs_pos, lbls_pos = self.get_batch(self.current_training_disease, which_batch="pos")

                disease_idx = self.TRAIN_DISEASES.index(self.current_training_disease)
                classifier = self.net_lgs[self.current_training_disease]

                task_code = self.latent_z_task[self.current_training_disease].to(self.device)
                _, neg_masks = self.net_g(imgs_neg, task_code)
                _, pos_masks = self.net_g(imgs_pos, task_code)

                masks_all = torch.cat((neg_masks, pos_masks))
                lbls_all = torch.cat((lbls_neg, lbls_pos))
                pred_all = classifier(masks_all)
                classifiers_loss = self.lgs_loss(pred_all.squeeze(), lbls_all[:, disease_idx])
                classifiers_loss = classifiers_loss * self.lambda_3

                optimizer = self.optim_lgs[self.current_training_disease]
                optimizer.zero_grad()
                classifiers_loss.backward()
                optimizer.step()

                classifier_iterations += 1
                print("classifier_iterations", classifier_iterations)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # every sample_step, visualze masks of the samples
            if gen_iterations % self.sample_step == 0:
                self.set_mode(mode='eval')
                with torch.no_grad():
                    self.save_disease_masks(gen_iterations)
                self.set_mode(mode='train')

            # every model_valid_step, do validation
            if gen_iterations % self.model_valid_step == 0:
                print(" - model validation-")
                auc, _, _ = self.validation(max_batches=500)
                valid_auc.append(auc)
                print("validation AUC: ", auc)
                print("current best AUC: ", best_auc)
                print("AUC of all epochs: ", valid_auc)

                if best_auc < auc:
                    best_auc = auc
                    logscalar("best_valid_auc", best_auc)
                    logscalar("best_valid_auc step", gen_iterations)
                    self.save_checkpoint(gen_iterations, is_best=True)
                    self.vis_classcenter(gen_iterations)

                else:
                    self.save_checkpoint(gen_iterations, is_best=False)
                self.set_mode(mode='train')


    def validation(self, max_batches):
        """
        Validation during training.
        """
        self.set_mode(mode='eval')
        valid_pred = []
        valid_true = []
        for c in range(self.num_class):
            valid_pred_c = []
            valid_true_c = []
            valid_pred.append(valid_pred_c)
            valid_true.append(valid_true_c)
        mean_classifier_loss = 0

        with torch.no_grad():
            steps = 0
            classifiers_loss = 0
            diter = iter(self.valid_loader)
            for i in tqdm(range(len(self.valid_loader))):
                data = next(diter)
                valid_data, valid_labels = data["img"], data["label"]
                valid_data = valid_data.to(self.device)
                valid_labels = valid_labels.to(self.device)
                for disease in self.TRAIN_DISEASES:
                    idx = self.TRAIN_DISEASES.index(disease)
                    classifier = self.net_lgs[disease]
                    task_code = self.latent_z_task[disease].to(self.device)
                    _, masks = self.net_g(valid_data, task_code)

                    y_pred_logits = classifier(masks)
                    y_pred = torch.sigmoid(y_pred_logits).squeeze()
                    valid_loss = self.lgs_loss(y_pred_logits.squeeze(), valid_labels[:, idx])

                    valid_pred[idx].append(to_numpy(y_pred))
                    valid_true[idx].append(to_numpy(valid_labels[:, idx]))
                    classifiers_loss += valid_loss * self.lambda_3
                steps += 1
                if max_batches is not None and steps >= max_batches:
                    break
            mean_classifier_loss = mean_classifier_loss/(self.num_class * steps)

        valid_class_auc = np.zeros(self.num_class)
        for c in range(self.num_class):
            valid_true_c = np.concatenate(valid_true[c])
            valid_pred_c = np.concatenate(valid_pred[c])
            valid_auc_mean = roc_auc_score(valid_true_c, valid_pred_c)
            valid_class_auc[c] = valid_auc_mean

        valid_mean_auc = np.mean(valid_class_auc)

        logdict = {
            "valid_cls_loss": mean_classifier_loss,
            "valid_mean_AUC": valid_mean_auc,
        }
        for n, v in logdict.items():
            logscalar(n, v)
        for i, l in enumerate(self.TRAIN_DISEASES):
            logscalar(l + "_AUC", valid_class_auc[i])

        return valid_mean_auc, valid_true, valid_pred


    def test(self, which_loader="test", save_result=False, result_dir=None):
        """
        Test the model. When which_loader is "test", test on the test set. When which_loader is "valid", test on the validation set.
        Test on validation set in order to get the best threshold for each disease.
        When which_loader is "testBB", test on the samples that have bounding box.
        testBB set with bounding box is small and some trained diseases do not have positive samples in this set, therefore treatly differently when compute auc.

        """

        if which_loader == "test":
            data_loader = self.test_loader
        if which_loader == "valid":
            data_loader = self.valid_loader
        if which_loader == "testBB":
            data_loader = self.dataloaders["BB_test"]

        self.set_mode(mode='eval')
        test_pred = []
        test_true = []

        for c in range(self.num_class):
            test_pred_c = []
            test_true_c = []
            test_pred.append(test_pred_c)
            test_true.append(test_true_c)

        with torch.no_grad():
            diter = iter(data_loader)
            for i in tqdm(range(len(data_loader))):
                data = next(diter)
                test_data, test_labels = data["img"], data["label"]
                test_data = test_data.to(self.device)
                test_labels = test_labels.to(self.device)

                for disease in self.TRAIN_DISEASES:
                    classifier = self.net_lgs[disease]
                    task_code = self.latent_z_task[disease].to(self.device)
                    _, masks = self.net_g(test_data, task_code)

                    y_pred_logits = classifier(masks)
                    y_pred = torch.sigmoid(y_pred_logits).squeeze()

                    idx = self.TRAIN_DISEASES.index(disease)
                    test_pred[idx].append(to_numpy(y_pred))
                    test_true[idx].append(to_numpy(test_labels[:, idx]))

            if which_loader == "testBB":
                pred = []
                true = []
                for c in range(self.num_class):
                    if len(test_true_c) > 0:
                        test_true_c = np.concatenate(test_true[c], axis=0)
                    else:
                        test_true_c = []
                    if len(test_pred_c) > 0:
                        test_pred_c = np.concatenate(test_pred[c], axis=0)
                    else:
                        test_pred_c = []
                    pred.append(test_pred_c)
                    true.append(test_true_c)

            else:
                test_class_auc = np.zeros(self.num_class)
                pred = []
                true = []
                for c in range(self.num_class):
                    test_true_c = np.concatenate(test_true[c])
                    test_pred_c = np.concatenate(test_pred[c])
                    pred.append(test_pred_c)
                    true.append(test_true_c)
                    test_auc_mean = roc_auc_score(test_true_c, test_pred_c)
                    test_class_auc[c] = test_auc_mean

                test_auc_mean = np.mean(test_class_auc)
                print("test_mean_AUC: " + str(test_auc_mean))
                for c in range(self.num_class):
                    print(self.TRAIN_DISEASES[c] + " auc: ", test_class_auc[c])

        pred = np.asarray(pred).T
        true = np.asarray(true).T

        if save_result:
            pred_path = os.path.join(result_dir, which_loader+'_pred.txt')
            true_path = os.path.join(result_dir, which_loader+'_true.txt')
            np.savetxt(pred_path, pred)
            np.savetxt(true_path, true)

        if which_loader == "testBB":
            return pred, true
        else:
            return pred, true, test_auc_mean


    def get_optimal_thresholds(self, save_result=True, result_dir=None):
        """
        Get the optimal thresholds for each disease on the validation set.
        Select the threshold that maximizes the sum of sensitivity and specificity (Youden index).
        """
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
            # Compute Youden index.
            sum = sensitivity + specificity
            best_t = statics['threshold'][np.argmax(sum)]
            best_threshold[disease] = best_t
            threshold.append(best_t)
        threshold = np.asarray(threshold)
        # Save the best threshold to avoid re-computing.
        if save_result:
            path = os.path.join(result_dir,'best_threshold.txt')
            np.savetxt(path, threshold)

        return best_threshold

    def get_attributes(self, input, label_idx):
        """
        Get the attribute map of the given input image with respect to a specific disease.
        """
        disease = self.TRAIN_DISEASES[label_idx]
        task_code = self.latent_z_task[disease].to(self.device)
        input = input.to(self.device)
        attrs = self.net_g(input, task_code)[1]
        return attrs

    def get_probs(self, inputs, label_idx):
        """
        Get the probability of the given input image with respect to a specific disease.
        """
        disease = self.TRAIN_DISEASES[label_idx]
        classifier = self.net_lgs[disease]
        attrs = self.get_attributes(inputs, label_idx)
        pred_logits = classifier(attrs)
        prob = torch.sigmoid(pred_logits)
        return prob

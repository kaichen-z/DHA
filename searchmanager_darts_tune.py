'''----System----'''
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

import utils
from model import hpo_da 
from model import darts_sup_tune
from visualize import plot
try:
    from tqdm import tqdm
except:
    print('-----No Package In Tqdm-----')

class ArchTuneRunManager_Darts: 
    def __init__(self, opt, config, dataset_config, train_config, arch_config, GENOTYPE, net_crit, sl_data, device):
        print(GENOTYPE)
        self.opt = opt
        self.GENOTYPE = GENOTYPE
        if dataset_config['name'] in ['sport8','mit67','flowers102','imagenet']:
            use_aux = arch_config['aux_weight'] > 0.
            try:
                genotype = GENOTYPE
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(GENOTYPE)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        elif dataset_config['name'] in ['cifar10','cifar100']:
            use_aux = arch_config['aux_weight'] > 0.
            try:
                genotype = GENOTYPE
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(GENOTYPE)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        print(genotype)
        self.criterion = net_crit
        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.start_epoch = self.train_config["start_epoch"]
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        print('-----The model for tunning the model-----')
    def reload(self):
        if self.config["model_path"] != "":
            print('Here we load')
            self.super_net.module.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            self.optimizer_w.load_state_dict(self.sl_data.load(["optimizer_w"], self.config["model_path"]))
            self.start_epoch = int(self.config['model_path'])+1
            for i in range(self.start_epoch):
                self.lr_scheduler.step()   
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
        self.learning_rate = utils.AverageMeter()
        self.weight_decay = utils.AverageMeter()
        self.entropy = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_w = torch.optim.SGD(self.super_net.module.parameters(), self.train_config['w_lr'], \
            momentum=self.train_config['w_momentum'], weight_decay=self.train_config['w_weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, self.train_config['epochs'])
    def training(self, train_loader, valid_loader_ori, epoch):
        print('==========Training=========')
        self.init_record()
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        print('==================LR:',lr)
        self.super_net.module.drop_path_prob=self.arch_config['drop_path_prob']*epoch/self.train_config['epochs']
        self.super_net.train()
        for step, train_batch in enumerate(train_loader):
            train_im, train_lb = (x_.cuda() for x_ in train_batch)   
            train_hat, train_aux_logits = self.super_net(train_im)       
            train_loss = self.criterion(train_hat, train_lb)
            if self.arch_config['aux_weight'] > 0.:
                train_loss += self.arch_config['aux_weight'] * self.criterion(train_aux_logits, train_lb)
            self.optimizer_w.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(self.super_net.module.parameters(), self.train_config['w_grad_clip'])
            self.optimizer_w.step()
            train_prec1, train_prec5 = utils.accuracy(train_hat, train_lb, topk=(1, 5))
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            del train_loss
            del train_im, train_lb
            del train_hat, train_aux_logits
        self.sl_data.logger.info('-Nor-ep:[{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-Nor-Con ep: [{}], Lr: {:.4f}'.format(
                        epoch, lr))
    def testing(self, testing_loader, epoch):
        self.super_net.eval()
        for step, test_batch in enumerate(testing_loader):
            test_im, test_lb = (x_.cuda() for x_ in test_batch)
            test_hat, test_aux_logits = self.super_net(test_im)
            test_loss = self.criterion(test_hat, test_lb)  
            test_prec1, test_prec5 = utils.accuracy(test_hat, test_lb, topk=(1, 5))
            self.test_losses.update(test_loss.item(), test_im.size(0))
            self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-Nor-ep:[{}], Test_l: {:.4f}, Test_Pr1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-Nor-Cur ep: [{}], Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            super_net = self.super_net
            self.sl_data.save('super_net', epoch)
            self.sl_data.logger.info("-Nor-Mode:{}, geno = {}".format(self.opt.Running, self.GENOTYPE))
            del super_net
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)

class ArchTuneRunManager_Darts_Da: 
    def __init__(self, opt, config, da_config, dataset_config, train_config, arch_config, GENOTYPE, net_crit, sl_data, device):
        print(GENOTYPE)
        self.opt = opt
        self.GENOTYPE = GENOTYPE
        self.da_search = hpo_da.DA_INIT(da_config, device=device)
        if dataset_config['name'] in ['sport8','mit67','flowers102','imagenet']:
            use_aux = arch_config['aux_weight'] > 0.
            try:
                genotype = GENOTYPE
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(GENOTYPE)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        elif dataset_config['name'] in ['cifar10','cifar100']:
            use_aux = arch_config['aux_weight'] > 0.
            try:
                genotype = GENOTYPE
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(GENOTYPE)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        print(genotype)
        self.criterion = net_crit
        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.start_epoch = self.train_config["start_epoch"]
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        print('-----The model for tunning the model-----')
    def reload(self):
        if self.config["model_path"] != "":
            print('Here we load')
            self.super_net.module.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            self.optimizer_w.load_state_dict(self.sl_data.load(["optimizer_w"], self.config["model_path"]))
            self.start_epoch = int(self.config['model_path'])+1
            for i in range(self.start_epoch):
                self.lr_scheduler.step()   
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
        self.learning_rate = utils.AverageMeter()
        self.weight_decay = utils.AverageMeter()
        self.entropy = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_w = torch.optim.SGD(self.super_net.module.parameters(), self.train_config['w_lr'], \
            momentum=self.train_config['w_momentum'], weight_decay=self.train_config['w_weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, self.train_config['epochs'])
        self.optimizer_da = torch.optim.SGD([self.da_search.policy_alpha.module], lr=self.da_search.config["d_lr"], momentum=self.da_search.config["d_momentum"], 
                        nesterov=self.da_search.config["d_nesterov"], weight_decay=self.da_search.config['d_weight_decay']) 
    def training(self, train_loader, valid_loader_ori, epoch):
        print('==========Training=========')
        self.init_record()
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        print('==================LR:',lr)
        train_loader.dataset.transform.choice_weights = self.da_search.weights_matrix()
        self.super_net.module.drop_path_prob=self.arch_config['drop_path_prob']*epoch/self.train_config['epochs']
        self.super_net.train()
        for step, train_batch in enumerate(train_loader):
            train_im, train_lb, train_ops = (x_.cuda() for x_ in train_batch)   
            train_hat, train_aux_logits = self.super_net(train_im)       
            train_loss = self.criterion(train_hat, train_lb)
            if self.arch_config['aux_weight'] > 0.:
                train_loss += self.arch_config['aux_weight'] * self.criterion(train_aux_logits, train_lb)
            train_loss_tf = self.da_search.prob_loss(-train_loss.detach(), train_ops)
            self.optimizer_da.zero_grad()
            train_loss_tf.backward() 
            #print('------------------data_augment------------------', self.da_search.policy_alpha.module.grad.sum())
            self.optimizer_da.step()
            self.optimizer_w.zero_grad()
            train_loss.mean().backward()
            nn.utils.clip_grad_norm_(self.super_net.module.parameters(), self.train_config['w_grad_clip'])
            self.optimizer_w.step()
            train_prec1, train_prec5 = utils.accuracy(train_hat, train_lb, topk=(1, 5))
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            del train_loss
            del train_im, train_lb, train_ops
            del train_hat, train_aux_logits
        self.sl_data.logger.info('-DA-ep:[{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-DA-Con ep: [{}], Lr: {:.4f}'.format(
                        epoch, lr))
        self.sl_data.logger.info('-DA-DA_entropy: {:.6}'.format(self.da_search.entropy_alpha()))
    def testing(self, testing_loader, epoch):
        self.super_net.eval()
        for step, test_batch in enumerate(testing_loader):
            test_im, test_lb, _ = (x_.cuda() for x_ in test_batch)
            test_hat, test_aux_logits = self.super_net(test_im)
            test_loss = self.criterion(test_hat, test_lb)  
            test_prec1, test_prec5 = utils.accuracy(test_hat, test_lb, topk=(1, 5))
            self.test_losses.update(test_loss.mean().item(), test_im.size(0))
            self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-DA-ep:[{}], Test_l: {:.4f}, Test_Pr1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-DA-Cur ep: [{}], -DA-Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            super_net = self.super_net
            self.sl_data.save('super_net', epoch)
            self.sl_data.logger.info("-DA-Mode:{}, geno = {}".format(self.opt.Running, self.GENOTYPE))
            del super_net
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)

class ArchTuneRunManager_Darts_Hpo: 
    def __init__(self, opt, config, hpo_config, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device):
        self.opt = opt#
        print(Genotype)
        self.GENOTYPE = Genotype
        self.hpo_search = hpo_da.HPO_INIT(hpo_config, device=device)
        use_aux = arch_config['aux_weight'] > 0.
        if dataset_config['name'] in ['sport8','mit67','flowers102','imagenet']:
            try:
                genotype = Genotype
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(Genotype)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        elif dataset_config['name'] in ['cifar10','cifar100']:
            try:
                genotype = Genotype
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(Genotype)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        self.criterion = net_crit
        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.start_epoch = self.train_config["start_epoch"]
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        print('-----The model for tunning the model-----')
    def reload(self):
        if self.config["model_path"] != "":
            print('Here we load')
            self.super_net.module.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            hpo_da_dict = self.sl_data.load(["hpo_da_dict"], self.config["model_path"])
            self.hpo_search.learning.module.data = hpo_da_dict['learning']
            self.hpo_search.weight_decay.module.data = hpo_da_dict['weight_decay']
            self.optimizer_hp.load_state_dict(self.sl_data.load(["optimizer_hp"], self.config["model_path"]))
            self.start_epoch = int(self.config['model_path'])+1
            #for i in range(self.start_epoch):
            #    self.lr_scheduler.step()   
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
        self.learning_rate = utils.AverageMeter()
        self.weight_decay = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_hp = torch.optim.SGD([self.hpo_search.learning.module, self.hpo_search.weight_decay.module], lr=self.hpo_search.config["h_lr"], 
                        momentum=self.hpo_search.config["h_momentum"], nesterov=self.hpo_search.config["h_nesterov"], weight_decay=self.hpo_search.config['h_weight_decay']) 
    def training(self, train_loader, valid_loader_ori, epoch):
        self.init_record()
        self.super_net.train()
        for step, (train_batch, val_batch_ori) in enumerate(zip(train_loader, valid_loader_ori)):
        #for step, (train_batch, val_batch_ori) in enumerate(zip(train_loader, valid_loader_ori)):
            train_im, train_lb = (x_.cuda() for x_ in train_batch)
            val_im, val_lb = (x_.cuda() for x_ in val_batch_ori) 
            train_hat, train_aux_logits = self.super_net(train_im)       
            train_loss = self.criterion(train_hat, train_lb)
            if self.arch_config['aux_weight'] > 0.:
                train_loss += self.arch_config['aux_weight'] * self.criterion(train_aux_logits, train_lb)
            # phase 2. The updating of the network's parameters.
            self.hpo_search.zero_grad(self.super_net.module.parameters())
            train_loss.mean().backward()
            nn.utils.clip_grad_norm_(self.super_net.module.parameters(), self.train_config['w_grad_clip'])
            self.hpo_search.SGD_STEP(self.super_net.module.named_parameters(), self.train_config['w_momentum']) 
            # phase 3. The updating process for the hyper-parameters.
            #val_im, val_lb = get_sample(val_im_ori, val_lb_ori, samples=val_im_ori.size(0)//4)
            val_hat, val_aux_logits = self.super_net(val_im)
            val_loss = self.criterion(val_hat, val_lb)    
            if self.arch_config['aux_weight'] > 0.:
                val_loss += self.arch_config['aux_weight'] * self.criterion(val_aux_logits, val_lb).mean()              
            self.optimizer_hp.zero_grad()
            val_loss.mean().backward()
            self.optimizer_hp.step()
            self.hpo_search.limited(freeze=True)
            self.hpo_search.reset_model(self.super_net.module.parameters())
            train_prec1, train_prec5 = utils.accuracy(train_hat, train_lb, topk=(1, 5))
            val_prec1, val_prec5 = utils.accuracy(val_hat, val_lb, topk=(1, 5))
            
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            self.learning_rate.update(self.hpo_search.learning.module.data, 1)
            self.weight_decay.update(self.hpo_search.weight_decay.module.data, 1)
            del train_loss, val_loss
            del train_hat, train_aux_logits, val_hat, val_aux_logits
            del train_im, train_lb, val_im, val_lb
        self.sl_data.logger.info('-HPO-ep:[{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-HPO-ep:[{}], Valid_l: {:.4f}, Valid_Pr1@: {:.2%}'.format(
                        epoch, self.val_losses.avg, self.val_top1.avg))
        self.sl_data.logger.info('-HPO-- LR: {:.6f}, -- W_Decay: {:.6f}'.format(
                        self.learning_rate.avg, self.weight_decay.avg))
    def testing(self, test_loader, epoch):
        self.super_net.eval()
        for step, test_batch in enumerate(test_loader):
            test_im, test_lb = (x_.cuda() for x_ in test_batch)     
            with torch.no_grad():
                test_hat, test_aux_logits = self.super_net(test_im)   
            test_loss = self.criterion(test_hat, test_lb)    
            test_prec1, test_prec5 = utils.accuracy(test_hat, test_lb, topk=(1, 5))
            self.test_losses.update(test_loss.mean().item(), test_im.size(0))
            self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-HPO-ep:[{}], Test_l: {:.4f}, Test_Pr1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-HPO-Cur ep: [{}], Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            super_net = self.super_net
            self.sl_data.save('super_net', epoch)
            self.sl_data.logger.info("-HPO-Mode:{}, geno = {}".format(self.opt.Running, self.GENOTYPE))
            del super_net
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)

class ArchTuneRunManager_Darts_Doal: 
    def __init__(self, opt, config, da_config, hpo_config, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device):
        self.opt = opt#
        print(Genotype)
        self.GENOTYPE = Genotype
        self.da_search = hpo_da.DA_INIT(da_config, device=device)
        self.hpo_search = hpo_da.HPO_INIT(hpo_config, device=device)
        use_aux = arch_config['aux_weight'] > 0.
        if dataset_config['name'] in ['sport8','mit67','flowers102','imagenet']:
            try:
                genotype = Genotype
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(Genotype)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkImageNet(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        elif dataset_config['name'] in ['cifar10','cifar100']:
            try:
                genotype = Genotype
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
            except:
                genotype = darts_sup_tune.Genotype_transfer(Genotype)
                self.super_net =  nn.DataParallel(darts_sup_tune.NetworkCIFAR(arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'], use_aux, genotype).to(device))
                self.super_net.module.drop_path_prob = arch_config['drop_path_prob']
        self.criterion = net_crit
        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.start_epoch = self.train_config["start_epoch"]
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        print('-----The model for tunning the model-----')

    def reload(self):
        if self.config["model_path"] != "":
            print('Here we load')
            self.super_net.module.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            hpo_da_dict = self.sl_data.load(["hpo_da_dict"], self.config["model_path"])
            self.hpo_search.learning.module.data = hpo_da_dict['learning']
            self.hpo_search.weight_decay.module.data = hpo_da_dict['weight_decay']
            self.optimizer_hp.load_state_dict(self.sl_data.load(["optimizer_hp"], self.config["model_path"]))
            self.start_epoch = int(self.config['model_path'])+1

    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
        self.learning_rate = utils.AverageMeter()
        self.weight_decay = utils.AverageMeter()

    def build_optimizer(self):
        self.optimizer_hp = torch.optim.SGD([self.hpo_search.learning.module, self.hpo_search.weight_decay.module], lr=self.hpo_search.config["h_lr"], 
                        momentum=self.hpo_search.config["h_momentum"], nesterov=self.hpo_search.config["h_nesterov"], weight_decay=self.hpo_search.config['h_weight_decay']) 
        self.optimizer_da = torch.optim.SGD([self.da_search.policy_alpha.module], lr=self.da_search.config["d_lr"], momentum=self.da_search.config["d_momentum"], 
                        nesterov=self.da_search.config["d_nesterov"], weight_decay=self.da_search.config['d_weight_decay'])     

    def training(self, train_loader, valid_loader_ori, epoch):
        self.init_record()
        self.super_net.train()
        for step, (train_batch, val_batch_ori) in enumerate(zip(train_loader, valid_loader_ori)):
        #for step, (train_batch, val_batch_ori) in enumerate(zip(train_loader, valid_loader_ori)):
            train_im, train_lb, train_ops = (x_.cuda() for x_ in train_batch)
            val_im, val_lb, val_ops = (x_.cuda() for x_ in val_batch_ori) 
            train_hat, train_aux_logits = self.super_net(train_im)       
            train_loss = self.criterion(train_hat, train_lb)
            if self.arch_config['aux_weight'] > 0.:
                train_loss += self.arch_config['aux_weight'] * self.criterion(train_aux_logits, train_lb)
            
            # phase 1. 
            train_loss_tf = self.da_search.prob_loss(-train_loss.detach(), train_ops)
            self.optimizer_da.zero_grad()
            train_loss_tf.backward() 
            #print('------------------data_augment------------------', self.da_search.policy_alpha.module.grad.sum())
            self.optimizer_da.step()

            # phase 2. The updating of the network's parameters.
            self.hpo_search.zero_grad(self.super_net.module.parameters())
            train_loss.mean().backward()
            nn.utils.clip_grad_norm_(self.super_net.module.parameters(), self.train_config['w_grad_clip'])
            self.hpo_search.SGD_STEP(self.super_net.module.named_parameters(), self.train_config['w_momentum']) 
            
            # phase 3. The updating process for the hyper-parameters.
            #val_im, val_lb = get_sample(val_im_ori, val_lb_ori, samples=val_im_ori.size(0)//4)
            val_hat, val_aux_logits = self.super_net(val_im)
            val_loss = self.criterion(val_hat, val_lb)    
            if self.arch_config['aux_weight'] > 0.:
                val_loss += self.arch_config['aux_weight'] * self.criterion(val_aux_logits, val_lb).mean()              
            self.optimizer_hp.zero_grad()
            val_loss.mean().backward()
            self.optimizer_hp.step()
            self.hpo_search.limited(freeze=True)
            self.hpo_search.reset_model(self.super_net.module.parameters())
            train_prec1, train_prec5 = utils.accuracy(train_hat, train_lb, topk=(1, 5))
            val_prec1, val_prec5 = utils.accuracy(val_hat, val_lb, topk=(1, 5))
            
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            self.learning_rate.update(self.hpo_search.learning.module.data, 1)
            self.weight_decay.update(self.hpo_search.weight_decay.module.data, 1)
            del train_loss, val_loss
            del train_hat, train_aux_logits, val_hat, val_aux_logits
            del train_im, train_lb, val_im, val_lb
        self.sl_data.logger.info('-Doal-ep:[{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-Doal-ep:[{}], Val_l: {:.4f}, Val_Pr1@: {:.2%}'.format(
                        epoch, self.val_losses.avg, self.val_top1.avg))
        self.sl_data.logger.info('-Doal-Lr_Rate: {:.6f}, --Weight_decay: {:.6f}'.format(
                        self.learning_rate.avg, self.weight_decay.avg))
        self.sl_data.logger.info('-Doal-DA_entropy: {:.6}'.format(self.da_search.entropy_alpha()))
    def testing(self, test_loader, epoch):
        self.super_net.eval()
        for step, test_batch in enumerate(test_loader):
            test_im, test_lb, test_ops = (x_.cuda() for x_ in test_batch)     
            with torch.no_grad():
                test_hat, test_aux_logits = self.super_net(test_im)   
            test_loss = self.criterion(test_hat, test_lb)    
            test_prec1, test_prec5 = utils.accuracy(test_hat, test_lb, topk=(1, 5))
            self.test_losses.update(test_loss.mean().item(), test_im.size(0))
            self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-Doal-ep:[{}], Test_l: {:.4f}, Test_Pr1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-Doal-Cur ep: [{}], Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            super_net = self.super_net
            self.sl_data.save('super_net', epoch)
            self.sl_data.logger.info("-Doal-Mode:{}, geno = {}".format(self.opt.Running, self.GENOTYPE))
            del super_net
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)

def get_sample(matrix, matrix_lb,samples):
    import random
    range_list = [i for i in range(matrix.size(0))]
    samples_list = random.sample(range_list, samples)
    return matrix[samples_list], matrix_lb[samples_list]











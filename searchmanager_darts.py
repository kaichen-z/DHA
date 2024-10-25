import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from visualize import plot

from model import hpo_da
from model import darts_sup as supernet
from model import ista_sup
from model import ista_sup_single

try:
    from tqdm import tqdm
except:
    print('-----No Package In Tqdm-----')
    
'''--------------------Darts With Darts Search Space (No DA and HPO)--------------------'''
class ArchSearchRunManager_Darts: 
    def __init__(self, opt, config, dataset_config, train_config, arch_config, net_crit, sl_data, device):
        self.opt = opt
        self.device = device
        self.config = config
        if self.opt.Running == 'darts': 
            self.super_net = supernet.Supernet(arch_config['input_channels'], arch_config['init_channels'], arch_config['n_classes'], arch_config['layers'],
                                    net_crit, data_set=dataset_config['name'], device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
        elif self.opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
            self.super_net = ista_sup.Supernet_Itsa(train_config, dataset_config, arch_config, \
                device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
            self.super_net.initialization()
        elif self.opt.Running in ['ista_single_nor']:
            self.super_net = ista_sup_single.Supernet_Itsa(train_config, dataset_config, arch_config, \
                device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
            self.super_net.initialization()
        self.net_crit = net_crit
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.arch_config = arch_config
        self.architect = Architect(self.super_net, train_config['w_momentum'], train_config['w_weight_decay'])
        self.sl_data = sl_data
        self.build_optimizer()
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        self.top_genotype = "Initial" # Recoding for best accuracy.
        self.start_epoch = train_config['start_epoch']
        self.reload()
    def reload(self):
        if self.config["model_path"] != "":
            self.super_net.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            self.optimizer_alpha.load_state_dict(self.sl_data.load(["optimizer_alpha"], self.config["model_path"]))
            self.optimizer_w.load_state_dict(self.sl_data.load(["optimizer_w"], self.config["model_path"]))
            self.start_epoch = int(self.config["model_path"])+1
            for i in range(self.start_epoch):
                self.lr_scheduler.step()
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_alpha = torch.optim.Adam(self.super_net.alphas(), self.arch_config['alpha_lr'], 
                                    betas=(0.5, 0.999),
                                    weight_decay=self.arch_config['alpha_weight_decay'])        
        self.optimizer_w = torch.optim.SGD(self.super_net.weights(), self.train_config['w_lr'], 
                                momentum=self.train_config['w_momentum'],
                                weight_decay=self.train_config['w_weight_decay'])
        if self.opt.Running in ['ista_single_nor']:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, self.train_config['retrain_epochs'], eta_min=self.train_config['w_lr_min'])
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, self.train_config['epochs'], eta_min=self.train_config['w_lr_min'])
    def training(self, train_loader, train_meta_loader, epoch):
        print('------------------Epoch------------------:', epoch)
        self.init_record()
        if 'all_freeze' not in self.config.keys(): # For darts
            self.lr_scheduler.step()
        elif 'all_freeze' in self.config.keys() and self.config['all_freeze']: # For frozen part 
            self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        print('==================LR:',lr)
        if self.opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
            self.super_net.pretrain() # This one works well for searching and tunning.
        if self.opt.Running in ['ista_single_nor']:
            self.super_net.pretrain(self.config['all_freeze'], drop_path_prob=self.arch_config['drop_path_prob'],\
                 i=epoch, epochs=self.train_config["retrain_epochs"])
        for _, (train_batch, val_batch) in enumerate(zip(train_loader, train_meta_loader)):
            train_im, train_lb = (x_.cuda() for x_ in train_batch)  
            val_im, val_lb = (x_.cuda() for x_ in val_batch)  
            if self.opt.Running == 'darts':
                self.optimizer_alpha.zero_grad()
                self.architect.unrolled_backward(train_im, train_lb, val_im, val_lb, lr, self.optimizer_w)
                #print('------------------alpha------------------', sum([i.grad.sum() for i in self.super_net.alpha_normal]))
                self.optimizer_alpha.step()
                self.optimizer_w.zero_grad()
                train_hat = self.super_net(train_im)
                train_loss = self.super_net.criterion(train_hat, train_lb)      
                train_prec1, train_prec5 = utils.accuracy(train_hat, train_lb, topk=(1, 5))
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
                #print('------------------weights------------------', sum([weights.grad.sum() for weights in self.super_net.weights()]))
                self.optimizer_w.step()
                val_hat = self.super_net(val_im)
                val_loss = self.super_net.criterion(val_hat, val_lb)   
                val_prec1, val_prec5 = utils.accuracy(val_hat, val_lb, topk=(1, 5))
            elif self.opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
                self.super_net.model.train()
                # The validation loss function update for the architecture parameters.
                self.optimizer_alpha.zero_grad()
                val_scores = self.super_net(val_im)
                val_scores = val_scores
                val_prec1, val_prec5 = utils.accuracy(val_scores, val_lb, topk=(1, 5))
                val_loss = F.cross_entropy(val_scores, val_lb)
                val_loss.backward(retain_graph=True)
                #print('Alphas', sum([i.grad.sum() for i in self.super_net.alphas()]))
                self.optimizer_alpha.step()
                # The training loss function update for the architecture parameters.
                self.optimizer_w.zero_grad()
                train_scores = self.super_net(train_im)
                train_scores = train_scores
                train_prec1, train_prec5 = utils.accuracy(train_scores, train_lb, topk=(1, 5))
                train_loss = F.cross_entropy(train_scores, train_lb)
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
                self.optimizer_w.step()
            elif self.opt.Running in ['ista_single_nor']:
                # There is no validation loss.
                self.super_net.model.train()
                self.optimizer_w.zero_grad()
                self.optimizer_alpha.zero_grad()
                train_scores, train_scores_aux = self.super_net(train_im, self.config['all_freeze'])
                train_prec1, train_prec5 = utils.accuracy(train_scores, train_lb, topk=(1, 5))
                train_loss = F.cross_entropy(train_scores, train_lb)
                if self.arch_config['auxiliary']:
                    train_loss_aux = F.cross_entropy(train_scores_aux, train_lb)
                    train_loss += self.arch_config['aux_weight'] * train_loss_aux
                train_loss.backward()
                self.optimizer_alpha.step()
                nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
                self.optimizer_w.step()
                val_scores = train_scores
                val_prec1 = train_prec1
                val_prec5 = train_prec5
                val_loss = train_loss
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            del train_loss, val_loss, val_prec1, val_prec5
            try:
                del train_loss_aux, train_scores_aux, val_loss_aux, train_scores
            except:
                error = 1
        del train_loader, train_meta_loader
        if self.opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
            self.super_net.postrain(epoch)
        elif self.opt.Running in ['ista_single_nor']:
            self.super_net.postrain(self.config['all_freeze'], self.arch_config['steps'], epoch)
        self.sl_data.logger.info('-Nor-Tr ep: [{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-Nor-Val ep: [{}], Val_l: {:.4f}, Val_Pr1@: {:.2%}'.format(
                        epoch, self.val_losses.avg, self.val_top1.avg))
        self.sl_data.logger.info('-Nor-Con ep: [{}], Lr: {:.4f}'.format(
                        epoch, lr))
        if self.opt.Running in ['ista_single_nor']:
            self.sl_data.logger.info('-Nor-Param S: [{}], Diff of Z: {:.4f}'.format(
                            self.super_net.param_size, self.super_net.sum_dist))
    def testing(self, test_loader, epoch):
        self.super_net.model.eval()
        self.super_net.model.drop_path_prob = 0
        #print('@@@@@1:', self.super_net)
        #print('@@@@@2:', self.super_net.model)
        with torch.no_grad():
            for _, test_batch in enumerate(test_loader):
                test_im, test_lb = (x_.cuda() for x_ in test_batch)     
                if self.opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
                    test_hat = self.super_net(test_im)   
                elif self.opt.Running in ['ista_single_nor']:
                    test_hat, _ = self.super_net(test_im, self.config['all_freeze'])   
                test_loss = F.cross_entropy(test_hat, test_lb)    
                test_prec1, _ = utils.accuracy(test_hat, test_lb, topk=(1, 5))
                self.test_losses.update(test_loss.mean().item(), test_im.size(0))
                self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-Nor-ep:[{}], Test_loss: {:.4f}, Test_Pr1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-Nor-Cur ep: [{}], Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            genotype = self.super_net.genotype()
            self.top_genotype = genotype
            plot_path_normal = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "normal")
            plot_path_reduce = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "reduce")
            caption = "Epoch {}".format(epoch+1)
            self.sl_data.logger.info("-Nor-Mode:{}, geno = {}".format(self.opt.Running, genotype))
            optimizer_alpha = self.optimizer_alpha
            optimizer_w = self.optimizer_w
            super_net = self.super_net
            if self.opt.Running != 'ista_single_nor':
                self.sl_data.save('super_net, optimizer_alpha, optimizer_w', epoch)
            else:
                alpha_normal = self.super_net.model.alphas_normal
                alpha_reduce = self.super_net.model.alphas_reduce
                self.sl_data.save('super_net,optimizer_alpha,optimizer_w,alpha_normal,alpha_reduce', epoch)
            del super_net, optimizer_alpha, optimizer_w

'''--------------------Darts With Darts Search Space (Data Augmentation)--------------------'''
class ArchSearchRunManager_Darts_Da: 
    def __init__(self, opt, config, da_config, dataset_config, train_config, arch_config, net_crit, sl_data, device):
        self.opt = opt
        self.device = device
        self.config = config
        self.da_search = hpo_da.DA_INIT(da_config, device=device)
        self.super_net = ista_sup_single.Supernet_Itsa(train_config, dataset_config, arch_config, \
            device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
        self.super_net.initialization()
        self.net_crit = net_crit
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.arch_config = arch_config
        self.architect = Architect(self.super_net, train_config['w_momentum'], train_config['w_weight_decay'])
        self.sl_data = sl_data
        self.build_optimizer()
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        self.top_genotype = "Initial" # Recoding for best accuracy.
        self.start_epoch = train_config['start_epoch']
        self.reload()
    def reload(self):
        if self.config["model_path"] != "":
            self.super_net.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            self.optimizer_alpha.load_state_dict(self.sl_data.load(["optimizer_alpha"], self.config["model_path"]))
            self.optimizer_w.load_state_dict(self.sl_data.load(["optimizer_w"], self.config["model_path"]))
            self.start_epoch = int(self.config["model_path"])+1
            for i in range(self.start_epoch):
                self.lr_scheduler.step()
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_alpha = torch.optim.Adam(self.super_net.alphas(), self.arch_config['alpha_lr'], 
                                    betas=(0.5, 0.999),
                                    weight_decay=self.arch_config['alpha_weight_decay'])        
        self.optimizer_w = torch.optim.SGD(self.super_net.weights(), self.train_config['w_lr'], 
                                momentum=self.train_config['w_momentum'],
                                weight_decay=self.train_config['w_weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_w, self.train_config['retrain_epochs'], eta_min=self.train_config['w_lr_min'])
        self.optimizer_da = torch.optim.SGD([self.da_search.policy_alpha.module], lr=self.da_search.config["d_lr"], momentum=self.da_search.config["d_momentum"], 
            nesterov=self.da_search.config["d_nesterov"], weight_decay=self.da_search.config['d_weight_decay']) 
    def training(self, train_loader, train_meta_loader, epoch):
        print('------------------Epoch------------------:', epoch)
        self.init_record()
        if self.config['all_freeze']: # For frozen part 
            self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        print('==================LR:',lr)
        self.super_net.pretrain(self.config['all_freeze'], drop_path_prob=self.arch_config['drop_path_prob'],\
                i=epoch, epochs=self.train_config["retrain_epochs"])
        self.super_net.model.train()
        for step, (train_batch, val_batch) in enumerate(zip(train_loader, train_meta_loader)):
            train_im, train_lb, train_ops = (x_.cuda() for x_ in train_batch)
            val_im, val_lb, val_ops = (x_.cuda() for x_ in val_batch)  
            train_scores, train_scores_aux = self.super_net(train_im, self.config['all_freeze'])
            train_prec1, train_prec5 = utils.accuracy(train_scores, train_lb, topk=(1, 5))
            train_loss = F.cross_entropy(train_scores, train_lb)
            if self.arch_config['auxiliary']:
                train_loss_aux = F.cross_entropy(train_scores_aux, train_lb)
                train_loss += self.arch_config['aux_weight'] * train_loss_aux
            # Step. 1
            train_loss_tf = self.da_search.prob_loss(-train_loss.detach(), train_ops)
            self.optimizer_da.zero_grad()
            train_loss_tf.backward()
            self.optimizer_da.step()
            # Step. 2
            self.optimizer_w.zero_grad()
            self.optimizer_alpha.zero_grad()
            train_loss.backward()
            self.optimizer_alpha.step()
            nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
            self.optimizer_w.step()
            val_scores = train_scores
            val_scores_aux = train_scores_aux
            val_prec1 = train_prec1
            val_prec5 = train_prec5
            val_loss = train_loss
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            del train_loss, val_loss, train_prec1, train_prec5, val_prec1, val_prec5
            del train_loss_aux, train_scores_aux, train_scores
        del train_loader, train_meta_loader
        self.super_net.postrain(self.config['all_freeze'], self.arch_config['steps'], epoch)
        self.sl_data.logger.info('-DA-Train ep: [{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-DA-Param Size: [{}], Diff of Z: {:.4f}'.format(
                        self.super_net.param_size, self.super_net.sum_dist))
        self.sl_data.logger.info('-DA_entropy: {:.6}'.format(
                        self.da_search.entropy_alpha()))
    def testing(self, test_loader, epoch):
        self.super_net.model.eval()
        self.super_net.model.drop_path_prob = 0
        with torch.no_grad():
            for step, test_batch in enumerate(test_loader):
                test_im, test_lb, _ = (x_.cuda() for x_ in test_batch)     
                test_hat, _ = self.super_net(test_im, self.config['all_freeze'])   
                test_loss = F.cross_entropy(test_hat, test_lb)    
                test_prec1, test_prec5 = utils.accuracy(test_hat, test_lb, topk=(1, 5))
                self.test_losses.update(test_loss.mean().item(), test_im.size(0))
                self.test_top1.update(test_prec1.item(), test_im.size(0))
        self.sl_data.logger.info('-DA-ep:[{}], Test_l: {:.4f}, Test_Prec1@: {:.2%}'.format(
                        epoch, self.test_losses.avg, self.test_top1.avg))
        if self.test_top1.avg > self.top1:
            self.top1 = self.test_top1.avg
            self.top_epoch = epoch
        self.sl_data.logger.info('-DA-Current: [{}], Best ep: {:.4f}, Best Perf@: {:.2%}'.format(
                        epoch, self.top_epoch, self.top1))
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            genotype = self.super_net.genotype()
            self.top_genotype = genotype
            plot_path_normal = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "normal")
            plot_path_reduce = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "reduce")
            caption = "Epoch {}".format(epoch+1)
            self.sl_data.logger.info("-DA-Mode:{}, geno = {}".format(self.opt.Running, genotype))
            optimizer_alpha = self.optimizer_alpha
            optimizer_w = self.optimizer_w
            super_net = self.super_net
            alpha_normal = self.super_net.model.alphas_normal
            alpha_reduce = self.super_net.model.alphas_reduce
            self.sl_data.save('super_net,optimizer_alpha,optimizer_w,alpha_normal,alpha_reduce', epoch)
            del super_net, optimizer_alpha, optimizer_w, alpha_normal, alpha_reduce

'''--------------------Doal With Darts Search Space and Darts' Optimization--------------------'''
class ArchSearchRunManager_Darts_Hpo: 
    # Build optimization and combine four modules together. 
    def __init__(self, opt, config, hpo_config, dataset_config, train_config, arch_config, net_crit, sl_data, device):
        # Going to record 
        self.opt = opt
        self.hpo_search = hpo_da.HPO_INIT(hpo_config, device=device)
        self.super_net = ista_sup_single.Supernet_Itsa(train_config, dataset_config, arch_config, \
                device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
        self.super_net.initialization()
        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.net_crit = net_crit
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        self.top_genotype = "Initial" # Recoding for best accuracy.
        self.start_epoch = train_config['start_epoch']
        self.reload()
    def reload(self):
        if self.config["model_path"] != "":
            self.super_net.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            hpo_da_dict = self.sl_data.load(["hpo_da_dict"], self.config["model_path"])
            self.hpo_search.learning.module.data = hpo_da_dict['learning']
            self.hpo_search.weight_decay.module.data = hpo_da_dict['weight_decay']
            self.optimizer_hp.load_state_dict(self.sl_data.load(["optimizer_hp"], self.config["model_path"]))
            self.optimizer_alpha.load_state_dict(self.sl_data.load(["optimizer_alpha"], self.config["model_path"]))
            self.start_epoch = int(self.config["model_path"])+1
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_alpha = torch.optim.Adam(self.super_net.alphas(), self.arch_config['alpha_lr'], betas=(0.5, 0.999),
                                    weight_decay=self.arch_config['alpha_weight_decay'])
        self.optimizer_hp = torch.optim.SGD([self.hpo_search.learning.module, self.hpo_search.weight_decay.module], lr=self.hpo_search.config["h_lr"], 
                        momentum=self.hpo_search.config["h_momentum"], nesterov=self.hpo_search.config["h_nesterov"], weight_decay=self.hpo_search.config['h_weight_decay']) 
    def training(self, train_loader, train_meta_loader, epoch):
        print('------------------Epoch------------------:', epoch)
        self.init_record()
        self.super_net.pretrain(self.config['all_freeze'], drop_path_prob=self.arch_config['drop_path_prob'],\
                i=epoch, epochs=self.train_config["retrain_epochs"])
        #for step, (train_batch, val_batch) in enumerate(tqdm(zip(train_loader, train_meta_loader))):
        for step, (train_batch, val_batch) in enumerate(zip(train_loader, train_meta_loader)):
            self.super_net.model.train()
            # Data Transformation with batch units. 
            train_im, train_lb = (x_.cuda() for x_ in train_batch)
            val_im, val_lb = (x_.cuda() for x_ in val_batch) 
            # Reinitializing the optimizer.
            train_scores, train_scores_aux = self.super_net(train_im, self.config['all_freeze'])
            train_prec1, train_prec5 = utils.accuracy(train_scores, train_lb, topk=(1, 5))
            train_loss = self.net_crit(train_scores, train_lb)
            if self.arch_config['auxiliary']:
                train_loss_aux = self.net_crit(train_scores_aux, train_lb)
                train_loss += self.arch_config['aux_weight'] * train_loss_aux
            self.hpo_search.zero_grad(self.super_net.weights())
            self.optimizer_alpha.zero_grad()  
            train_loss.mean().backward()
            self.optimizer_alpha.step()    
            nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
            self.hpo_search.SGD_STEP(self.super_net.named_weights(), self.train_config['w_momentum'])
            val_scores, val_scores_aux = self.super_net(val_im, self.config['all_freeze'])
            val_prec1, val_prec5 = utils.accuracy(val_scores, val_lb, topk=(1, 5))
            val_loss = F.cross_entropy(val_scores, val_lb)
            self.optimizer_hp.zero_grad()
            val_loss.mean().backward() 
            #print('------------------Learning_grad------------------', self.hpo_search.learning.module.grad.sum())
            #print('------------------Weight_devay_grad------------------', self.hpo_search.weight_decay.module.grad.sum())
            self.optimizer_hp.step()
            #print('------------------Weight_devay------------------', self.hpo_search.weight_decay.module)
            self.hpo_search.limited(freeze=self.config['all_freeze'])
            self.hpo_search.reset_model(self.super_net.weights())
            # ---------- Recording Process ----------
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            del train_loss, val_loss
        self.super_net.postrain(self.config['all_freeze'], self.arch_config['steps'], epoch)
        self.sl_data.logger.info('-HPO-Tr ep: [{}], Tr_l: {:.4f}, Tr_Pr1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-HPO-Val ep: [{}], Val_l: {:.4f}, Val_Pr1@: {:.2%}'.format(
                        epoch, self.val_losses.avg, self.val_top1.avg))
        self.sl_data.logger.info('-HPO-Lr: {:.6f}, --Weight_decay: {:.6f}'.format(
                        self.hpo_search.learning.module.data,self.hpo_search.weight_decay.module.data))
    def testing(self, test_loader, epoch):
        self.super_net.model.eval()
        for step, test_batch in enumerate(test_loader):
            test_im, test_lb = (x_.cuda() for x_ in test_batch)     
            with torch.no_grad():
                test_hat, test_aux_logits = self.super_net(test_im, self.config['all_freeze'])  
            test_loss = self.net_crit(test_hat, test_lb)    
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
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            genotype = self.super_net.genotype()
            self.top_genotype = genotype
            plot_path_normal = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "normal")
            plot_path_reduce = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "reduce")
            caption = "Epoch {}".format(epoch+1)
            self.sl_data.logger.info("-HPO-Mode: {}, geno = {}".format(self.opt.Running, genotype))
            optimizer_hp = self.optimizer_hp
            optimizer_alpha= self.optimizer_alpha
            super_net = self.super_net
            hpo_da_dict = {'learning':self.hpo_search.learning.module.data, 
            'weight_decay':self.hpo_search.weight_decay.module.data}
            alpha_normal = self.super_net.model.alphas_normal
            alpha_reduce = self.super_net.model.alphas_reduce
            self.sl_data.save('super_net,optimizer_hp,optimizer_alpha,hpo_da_dict,alpha_normal,alpha_reduce', epoch)
            del super_net, optimizer_alpha, hpo_da_dict

'''--------------------Doal With Darts Search Space and Darts' Optimization--------------------'''
class ArchSearchRunManager_Darts_Doal: 
    # Build optimization and combine four modules together. 
    def __init__(self, opt, config, hpo_config, da_config, dataset_config, train_config, arch_config, net_crit, sl_data, device):
        # Going to record 
        self.opt = opt
        self.hpo_search = hpo_da.HPO_INIT(hpo_config, device=device)
        self.da_search = hpo_da.DA_INIT(da_config, device=device)
        
        self.super_net = ista_sup_single.Supernet_Itsa(train_config, dataset_config, arch_config, \
            device_ids=[int(i) for i in range(len(config['gpu'].split(',')))]).to(device)
        self.super_net.initialization()

        self.config = config
        self.dataset_config = dataset_config
        self.train_config =  train_config
        self.arch_config = arch_config
        self.net_crit = net_crit
        self.sl_data = sl_data
        self.device = device
        self.build_optimizer()
        self.top1 = 0 # Recoding for best accuracy.
        self.top_epoch = 0
        self.top_genotype = "Initial" # Recoding for best accuracy.
        self.start_epoch = train_config['start_epoch']
        self.reload()
    def reload(self):
        if self.config["model_path"] != "":
            self.super_net.load_state_dict(self.sl_data.load(["super_net"], self.config["model_path"]))
            hpo_da_dict = self.sl_data.load(["hpo_da_dict"], self.config["model_path"])
            self.hpo_search.learning.module.data = hpo_da_dict['learning']
            self.hpo_search.weight_decay.module.data = hpo_da_dict['weight_decay']
            self.da_search.policy_alpha.module.data = hpo_da_dict['policy_alpha']
            self.optimizer_hp.load_state_dict(self.sl_data.load(["optimizer_hp"], self.config["model_path"]))
            self.optimizer_da.load_state_dict(self.sl_data.load(["optimizer_da"], self.config["model_path"]))
            self.optimizer_alpha.load_state_dict(self.sl_data.load(["optimizer_alpha"], self.config["model_path"]))
            self.start_epoch = int(self.config["model_path"])+1
    def init_record(self):
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_losses = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_losses = utils.AverageMeter()
        self.test_top1 = utils.AverageMeter()
        self.test_losses = utils.AverageMeter()
    def build_optimizer(self):
        self.optimizer_hp = torch.optim.SGD([self.hpo_search.learning.module, self.hpo_search.weight_decay.module], lr=self.hpo_search.config["h_lr"], 
                        momentum=self.hpo_search.config["h_momentum"], nesterov=self.hpo_search.config["h_nesterov"], weight_decay=self.hpo_search.config['h_weight_decay']) 
        self.optimizer_da = torch.optim.SGD([self.da_search.policy_alpha.module], lr=self.da_search.config["d_lr"], momentum=self.da_search.config["d_momentum"], 
                        nesterov=self.da_search.config["d_nesterov"], weight_decay=self.da_search.config['d_weight_decay']) 
        self.optimizer_alpha = torch.optim.Adam(self.super_net.alphas(), self.arch_config['alpha_lr'], betas=(0.5, 0.999),
                                    weight_decay=self.arch_config['alpha_weight_decay'])
    def training(self, train_loader, train_meta_loader, epoch):
        print('------------------Epoch------------------:', epoch)
        self.init_record()
        self.super_net.train()
        if self.opt.Running in ['ista_single_doal']:
            self.super_net.pretrain(self.config['all_freeze'], drop_path_prob=self.arch_config['drop_path_prob'],\
                 i=epoch, epochs=self.train_config["retrain_epochs"])
        for step, (train_batch, val_batch) in enumerate(zip(train_loader, train_meta_loader)):
            #print('==========={}=========='.format(step))
            train_loader.dataset.transform.choice_weights = self.da_search.weights_matrix()
            # Data Transformation with batch units. 
            train_im, train_lb, train_ops = (x_.cuda() for x_ in train_batch)
            val_im, val_lb, val_ops = (x_.cuda() for x_ in val_batch) 
            # Reinitializing the optimizer.
            train_scores, train_scores_aux = self.super_net(train_im, self.config['all_freeze'])
            train_prec1, train_prec5 = utils.accuracy(train_scores, train_lb, topk=(1, 5))
            train_loss = self.net_crit(train_scores, train_lb)
            if self.arch_config['auxiliary']:
                train_loss_aux = self.net_crit(train_scores_aux, train_lb)
                train_loss += self.arch_config['aux_weight'] * train_loss_aux
            #Step.1
            train_loss_tf = self.da_search.prob_loss(-train_loss.detach(), train_ops)
            self.optimizer_da.zero_grad()
            train_loss_tf.backward() 
            #print('------------------Data_augment------------------', self.da_search.policy_alpha.module.grad.sum())
            #print('------------------Entropy------------------', self.da_search.entropy_alpha())
            self.optimizer_da.step()
            #Step.2
            self.hpo_search.zero_grad(self.super_net.weights())
            self.optimizer_alpha.zero_grad()  
            train_loss.mean().backward()
            self.optimizer_alpha.step()    
            nn.utils.clip_grad_norm_(self.super_net.weights(), self.train_config['w_grad_clip'])
            self.hpo_search.SGD_STEP(self.super_net.named_weights(), self.train_config['w_momentum'])
            #Step.3
            val_scores, val_scores_aux = self.super_net(val_im, self.config['all_freeze'])
            val_prec1, val_prec5 = utils.accuracy(val_scores, val_lb, topk=(1, 5))
            val_loss = F.cross_entropy(val_scores, val_lb)
            self.optimizer_hp.zero_grad()
            val_loss.mean().backward() 
            #print('------------------Learning_grad------------------', self.hpo_search.learning.module.grad.sum())
            #print('------------------Weight_devay_grad------------------', self.hpo_search.weight_decay.module.grad.sum())
            self.optimizer_hp.step()
            #print('------------------Weight_devay------------------', self.hpo_search.weight_decay.module)
            self.hpo_search.limited(freeze=self.config['all_freeze'])
            self.hpo_search.reset_model(self.super_net.weights())
            # ---------- Recording Process ----------
            self.train_losses.update(train_loss.mean().item(), train_im.size(0))
            self.train_top1.update(train_prec1.item(), train_im.size(0))
            self.train_top5.update(train_prec5.item(), train_im.size(0))
            self.val_losses.update(val_loss.mean().item(), val_im.size(0))
            self.val_top1.update(val_prec1.item(), val_im.size(0))
            self.val_top5.update(val_prec5.item(), val_im.size(0))
            del train_im, train_lb, train_ops
            del val_im, val_lb, val_ops
            del train_loss, val_loss
        if self.opt.Running in ['ista_single_doal']:
            self.super_net.postrain(self.config['all_freeze'], self.arch_config['steps'], epoch)
        self.sl_data.logger.info('-Doal-Tr ep: [{}], Tr_loss: {:.4f}, Tr_Prec1@: {:.2%}'.format(
                        epoch, self.train_losses.avg, self.train_top1.avg))
        self.sl_data.logger.info('-Doal-Val ep: [{}], Val_loss: {:.4f}, Val_Prec1@: {:.2%}'.format(
                        epoch, self.val_losses.avg, self.val_top1.avg))
        self.sl_data.logger.info('-Doal-Lr: {:.6f}, --Weight_decay: {:.6f}, --DA_entropy: {:.4}'.format(
                        self.hpo_search.learning.module.data,self.hpo_search.weight_decay.module.data, self.da_search.entropy_alpha()))
    def testing(self, test_loader, epoch):
        self.super_net.eval()
        for step, test_batch in enumerate(test_loader):
            test_im, test_lb, _ = (x_.cuda() for x_ in test_batch)     
            with torch.no_grad():
                if self.opt.Running in ['ista_single_doal']:
                    test_hat, test_aux_logits = self.super_net(test_im, self.config['all_freeze'])  
                else:
                    break
            test_loss = self.net_crit(test_hat, test_lb)    
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
        utils.written(epoch, self.test_top1.avg, self.sl_data.root)
        if epoch%self.config['save_freq']==0 or self.test_top1.avg >= self.config['best_acc']:
            genotype = self.super_net.genotype()
            self.top_genotype = genotype
            plot_path_normal = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "normal")
            plot_path_reduce = os.path.join(self.config['save_root'], self.dataset_config['name']+'_'+self.opt.Running+'_'+self.config["save_name"], f"epoch_{epoch}", "reduce")
            caption = "Epoch {}".format(epoch+1)
            self.sl_data.logger.info("-Doal-Mode: {}, geno = {}".format(self.opt.Running, genotype))
            optimizer_hp = self.optimizer_hp
            optimizer_da = self.optimizer_da
            optimizer_alpha= self.optimizer_alpha
            super_net = self.super_net
            hpo_da_dict = {'learning':self.hpo_search.learning.module.data, 'weight_decay':self.hpo_search.weight_decay.module.data, 
                'policy_alpha': self.da_search.policy_alpha.module.data}
            if self.opt.Running not in ['ista_single_doal']:
                self.sl_data.save('super_net,optimizer_hp,optimizer_alpha,hpo_da_dict', epoch)
            else:
                alpha_normal = self.super_net.model.alphas_normal
                alpha_reduce = self.super_net.model.alphas_reduce
                self.sl_data.save('super_net,optimizer_hp,optimizer_alpha,hpo_da_dict,alpha_normal,alpha_reduce', epoch)
            del super_net, optimizer_hp, optimizer_da, optimizer_alpha, hpo_da_dict

class Architect():
    # Alpha Update for Darts
    def __init__(self, net, w_momentum, w_weight_decay):
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)
        gradients = torch.autograd.grad(loss, self.net.weights())
        with torch.no_grad():
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        self.virtual_step(trn_X, trn_y, xi, w_optim)
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        hessian = self.compute_hessian(dw, trn_X, trn_y)
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h
    def compute_hessian(self, dw, trn_X, trn_y):
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

def Genotype_transfer(Genotype):
    from collections import namedtuple
    normal = Genotype.normal
    normal_cat = Genotype.normal_concat
    reduction = Genotype.reduce
    reduction_cat = Genotype.reduce_concat
    normal_new = []
    normal_cat_new = []
    reduction_new = []
    reduction_cat_new = []
    for i in normal:
        normal_new.append(i[0])
        normal_new.append(i[0])
    for j in reduction:
        reduction_new.append(j[0])
        reduction_new.append(j[0])
    Genotype_new = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    return Genotype_new(normal=normal_new, normal_concat=range(min(normal_cat),max(normal_cat)+1),
                        reduce=reduction_new, reduce_concat=range(min(reduction_cat),max(reduction_cat)+1))
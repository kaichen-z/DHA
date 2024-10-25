'''----System----'''
import os 
import sys
import random
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import namedtuple

from utils import *
from data import data_aug
from options import parse_arguments_darts 

from searchmanager_darts import ArchSearchRunManager_Darts
from searchmanager_darts import ArchSearchRunManager_Darts_Da
from searchmanager_darts import ArchSearchRunManager_Darts_Hpo
from searchmanager_darts import ArchSearchRunManager_Darts_Doal

from searchmanager_darts_tune import ArchTuneRunManager_Darts
from searchmanager_darts_tune import ArchTuneRunManager_Darts_Doal
from searchmanager_darts_tune import ArchTuneRunManager_Darts_Da
from searchmanager_darts_tune import ArchTuneRunManager_Darts_Hpo


def main(opt, conf):
    if torch.cuda.is_available:
        device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")
    random.seed(conf["seed"])
    os.environ['PYTHONHASHSEED'] = str(conf["seed"])
    np.random.seed(conf["seed"])
    torch.manual_seed(conf["seed"])
    torch.cuda.manual_seed(conf["seed"])
    torch.cuda.manual_seed_all(conf["seed"])
    torch.backends.cudnn.deterministic = True
    # -----------------------Useful Config-----------------------
    dataset_config = conf["dataset"] 
    train_config = conf["train"]
    arch_config = conf["train_architecture"] 
    hpo_config = conf["hpo"]
    da_config = conf["da"]
    # -----------------------Modifying Configuration Based On Choice.
    if opt.batch_size != 0:
        dataset_config["batch_size"] = int(opt.batch_size)
    if opt.lr != 0:
        train_config["w_lr"] = opt.lr        
    if opt.hpo_lr != 0:
        hpo_config["h_lr"] = opt.hpo_lr
    # -----------------------Recording-----------------------    
    save_name = dataset_config['name']+'_'+opt.Running+'_'+conf["save_name"]
    root_load = dataset_config['name']+'_'+opt.Running+'_'+conf["load_name"] 
    sl_data = IOdata(root = conf["save_root"], root_load = root_load, save_name = save_name)
    sl_data.logger.info(opt)    
    # -----------------------Data-----------------------    
    if dataset_config['name'] in ['cifar10','cifar100']: 
        print('----------{:s}-----------'.format(dataset_config['name']))
        # No story about Data augmentation part and Hyper-parameters
        if opt.Running in ['darts', 'ista_nor',\
            'ista_da','ista_hpo','ista_doal',\
            'ista_single_nor','ista_single_hpo']:
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_cifar(dataset_config)
        elif opt.Running in ['ista_single_da','ista_single_doal']: 
            train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude'])
        else:
            sys.stderr.write('This is the wrong matching.')
            raise SystemExit(1)
        
    elif dataset_config['name'] in ['sport8','mit67','flowers102']:
        print('----------{:s}-----------'.format(dataset_config['name']))
        if opt.Running in ['darts','ista_nor',\
        'ista_da','ista_hpo','ista_doal',\
        'ista_single_nor','ista_single_hpo']:
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_image224(dataset_config)
        elif opt.Running in ['ista_single_da','ista_single_doal']: 
            train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude'])
        else:
            sys.stderr.write('This is the wrong matching.')
            raise SystemExit(1)       

    elif dataset_config['name'] in ['imagenet']:  
        print('----------ImageNat-----------')
        if opt.Running in ['darts','ista_nor',\
            'ista_da','ista_hpo','ista_doal',\
            'ista_single_nor','ista_single_hpo']:
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_imagenet(dataset_config)
        elif opt.Running in ['ista_single_da','ista_single_doal']: 
            train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude'])
        else:
            sys.stderr.write('This is the wrong matching.')
            raise SystemExit(1)
    # -----------------------Running Manager-----------------------    
    if opt.Running in ['darts','ista_nor','ista_da','ista_hpo','ista_doal',\
            'ista_single_nor']:
        print(f'----------------------{opt.Running}----------------------')
        sl_data.logger.info(f'-----------{opt.Running}-----------')
        net_crit = nn.CrossEntropyLoss()    
        runmanager = ArchSearchRunManager_Darts(opt, conf, dataset_config, train_config, arch_config, net_crit, sl_data, device)   

    elif opt.Running in ['ista_single_da']:
        print(f'----------------------{opt.Running}----------------------')
        sl_data.logger.info(f'-----------{opt.Running}-----------')
        net_crit = nn.CrossEntropyLoss(reduction='none')   
        runmanager = ArchSearchRunManager_Darts_Da(opt, conf, da_config, dataset_config, train_config, arch_config, net_crit, sl_data, device)   
    
    elif opt.Running in ['ista_single_hpo']:
        print(f'----------------------{opt.Running}----------------------')
        sl_data.logger.info(f'-----------{opt.Running}-----------')
        net_crit = nn.CrossEntropyLoss()   
        runmanager = ArchSearchRunManager_Darts_Hpo(opt, conf, hpo_config, dataset_config, train_config, arch_config, net_crit, sl_data, device)   
    
    elif opt.Running in ['ista_single_doal']:
        print(f'----------------------{opt.Running}----------------------')
        sl_data.logger.info(f'-----------{opt.Running}-----------')
        net_crit = nn.CrossEntropyLoss(reduction='none')   
        runmanager = ArchSearchRunManager_Darts_Doal(opt, conf, hpo_config, da_config, dataset_config, train_config, arch_config, net_crit, sl_data, device)   
    # -----------------------Training-----------------------       
    print('-----Training_data-----:',len(train_data))
    print('-----Val_data-----:',len(val_data_ori))
    print('-----Test_data-----:',len(test_data))
    for epochs in range(runmanager.start_epoch, train_config['epochs']):
        runmanager.training(train_data, val_data_ori, epochs)
        if opt.Running in ['ista_single_nor',\
            'ista_single_da','ista_single_hpo',\
            'ista_single_doal'] and runmanager.super_net.Stop_flag:
            break

    sl_data.logger.info('------------------------The searching period is finished.------------------------')
    sl_data.logger.info(runmanager.super_net.genotype())

    print('------------------------The searching period is finished.------------------------')
    '''-----------Changing Configuration----------'''
    if opt.Running in ['ista_nor','ista_da','ista_hpo','ista_doal']:
        print('------------------------We move to the tunning type.------------------------')
        with open(opt.conf_path) as fid:
            conf = json.load(fid)
            dataset_config = conf["tune_dataset"] 
            train_config = conf["tune"] 
            arch_config = conf["tune_architecture"]
            hpo_config = conf["hpo_tune"]
            da_config = conf["da_tune"]    
    if opt.lr_tune != 0: # Only for Two-Stage ISTA
        train_config["w_lr"] = opt.lr_tune   
    try:
        if opt.lr_tune != 0:
            hpo_config["h_lr_tune"] = opt.hpo_lr_tune         
    except:
        error = 1

    '''-----------Changing Dataloader----------'''
    if opt.Running in ['ista_nor','ista_hpo','ista_single_nor','ista_single_hpo']:
        print('------------------------We move to the new dataloader.------------------------')
        if dataset_config['name'] in ['cifar10','cifar100']: 
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_cifar(dataset_config)
        elif dataset_config['name'] in ['sport8','mit67','flowers102']:
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_image224(dataset_config)
        elif dataset_config['name'] in ['imagenet']:  
            train_data, val_data_ori, test_data = data_aug.build_dataset_darts_imagenet(dataset_config)

    if  opt.Running == 'ista_nor':
        Genotype = runmanager.super_net.genotype()
        print('-----------Genotype-------------', Genotype)
        net_crit = nn.CrossEntropyLoss()   
        runmanager_retrain = ArchTuneRunManager_Darts(opt, conf, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device) 
        del runmanager
        for epochs in range(train_config['epochs']):
            runmanager_retrain.training(train_data, val_data_ori, epochs)

    elif opt.Running == 'ista_da': # There is no need to genralize it to the two stage 
        train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude'])
        Genotype = runmanager.super_net.genotype()
        print('-----------Genotype-------------', Genotype)
        net_crit = nn.CrossEntropyLoss(reduction='none')   
        runmanager_retrain = ArchTuneRunManager_Darts_Da(opt, conf, da_config, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device)
        del runmanager
        for epochs in range(train_config['epochs']):
            runmanager_retrain.training(train_data, val_data_ori, epochs)
                
    elif opt.Running == 'ista_hpo': # There is no need to genralize it to the two stage
        Genotype = runmanager.super_net.genotype()
        print('-----------Genotype-------------', Genotype)
        net_crit = nn.CrossEntropyLoss()     
        runmanager_retrain = ArchTuneRunManager_Darts_Hpo(opt, conf, hpo_config, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device)
        del runmanager
        for epochs in range(train_config['epochs']):
            runmanager_retrain.training(train_data, val_data_ori, epochs)

    elif opt.Running == 'ista_doal': # There is no need to genralize it to the two stage
        train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude'])
        Genotype = runmanager.super_net.genotype()
        print('-----------Genotype-------------', Genotype)
        net_crit = nn.CrossEntropyLoss(reduction='none')    
        runmanager_retrain = ArchTuneRunManager_Darts_Doal(opt, conf, da_config, hpo_config, dataset_config, train_config, arch_config, Genotype, net_crit, sl_data, device)
        del runmanager
        for epochs in range(train_config['epochs']):
            runmanager_retrain.training(train_data, val_data_ori, epochs)

    if  opt.Running == 'ista_single_nor':
        for epochs in range(train_config['retrain_epochs']):
            runmanager.config['all_freeze'] = True
            runmanager.training(train_data, val_data_ori, epochs)

    if opt.Running == 'ista_single_da':
        train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude_tune'])
        for param_group in runmanager.optimizer_da.param_groups:
            param_group['lr'] = da_config['d_lr_tune']
        print('-----------Genotype-------------', runmanager.super_net.genotype())
        for epochs in range(train_config['retrain_epochs']):
            conf['all_freeze'] = True
            runmanager.training(train_data, val_data_ori, epochs)

    if  opt.Running == 'ista_single_hpo':
        for param_group in runmanager.optimizer_hp.param_groups:
            param_group['lr'] = hpo_config['h_lr_tune']
        if hpo_config['restart']:        
            runmanager.hpo_search.restart()
        for epochs in range(train_config['retrain_epochs']):
            runmanager.config['all_freeze'] = True
            runmanager.training(train_data, val_data_ori, epochs)

    if opt.Running == 'ista_single_doal':
        train_data, _, _, val_data_ori, test_data = data_aug.build_dataset(dataset_config, opt, da_config['magnitude_tune'])
        for param_group in runmanager.optimizer_da.param_groups:
            param_group['lr'] = da_config['d_lr_tune']
        for param_group in runmanager.optimizer_hp.param_groups:
            param_group['lr'] = hpo_config['h_lr_tune']
        if hpo_config['restart']:        
            runmanager.hpo_search.restart()
        print('-----------Genotype-------------', runmanager.super_net.genotype())
        for epochs in range(train_config['retrain_epochs']):
            conf['all_freeze'] = True
            runmanager.training(train_data, val_data_ori, epochs)

if __name__ == '__main__':
    args, conf = parse_arguments_darts(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"] 
    for i in range(8):
        print(f'-----{i}------')
        try:
            print(torch.cuda.get_device_name(i))
        except:
            print('-----No Device Detected-----')
    main(args, conf)
    
    
    
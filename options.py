import argparse
from utils import *

'''--------------------------Darts Search Space--------------------------'''
import model.darts_genotypes as gt
def parse_arguments_darts(argv):
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--conf_path', type=str, default='conf/darts/cifar10_darts.json', choices= [\
        'conf/darts/cifar10_darts.json',\
        'conf/darts/imagenet_darts.json',\
        'conf/darts/imagenet_darts_dsnas.json',\
        'conf/ista/cifar10_ista.json',\
        'conf/ista/cifar10_ista_single.json',\
        'conf/ista/cifar10_ista_single_doal.json',\
        'conf/ista/cifar100_ista.json',\
        'conf/ista/cifar100_ista_single.json',\
        'conf/ista/cifar100_ista_single_doal.json',\
        'conf/ista/sport8_ista.json',\
        'conf/ista/sport8_ista_single.json',\
        'conf/ista/sport8_ista_single_doal.json',\
        'conf/ista/sport8_ista_single_doal2.json',\
        'conf/ista/sport8_ista_single_doal3.json',\
        'conf/ista/sport8_ista_single_doal4.json',\
        'conf/ista/mit67_ista.json',\
        'conf/ista/mit67_ista_single.json',\
        'conf/ista/flowers102_ista.json',\
        'conf/ista/flowers102_ista_single.json',\
        'conf/ista/imagenet_ista.json',\
        'conf/ista/imagenet_ista_single.json'])
    parser.add_argument('--da_choice', type=str, default="batch_transformation")
    parser.add_argument('--Running',type=str,default='darts',choices=[\
        'darts', \
        # darts: normal 2 level darts
        'ista_nor','ista_single_nor',\
        # ISTA searching with normal tunning
        # ISTA searching and tunning
        'ista_da','ista_single_da',\
        # ISTA searching with data aug tunning
        # ISTA searching with single data aug
        'ista_hpo','ista_single_hpo',\
        'ista_doal','ista_single_doal'])
    parser.add_argument('--genotype', type=str, default='image224',choices=['image32','image224','image256'])
    # Modification for optimization
    parser.add_argument('--hpo_lr', type=float, default=0)
    parser.add_argument('--hpo_lr_tune', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--lr_tune', type=float, default=0)
    
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--tqdm', type=bool, default=False)
    args, unknown_args = parser.parse_known_args(argv)
    clear_args(args, unknown_args) # Args is the overall dictionary, while unknown_args is written to the conf_path.
    conf = get_json(args.conf_path, unknown_args)
    print('---args---', args)
    print('---unknown_args---', unknown_args)
    print('---GPU---', conf["gpu"])
    print('---SAVE_NAME---', conf["save_name"])
    print('---SAVE_ROOT---', conf["save_root"])
    return args, conf

def clear_args(args, json_args):
    args_keys = [i for i in args.__dict__.keys()]
    i, j =(0,0)
    while i < len(json_args):
        for j in range(len(args_keys)):
            if args_keys[j] in json_args[i]:
                del json_args[i]
                i -= 1
        i += 1





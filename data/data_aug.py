import torchvision
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import datasets
import random
import os
import sys
import pickle
import numpy as np
from PIL import Image

from .auto_augment import *

'''------------------------------------------Data Augmentation For DA Optimization------------------------------------------'''
def build_dataset(dataset_conf, opt, magnitude):
    aa_params = dict(translate_const = int(dataset_conf["img_min_size"] * 0.45),
            img_mean = tuple([min(255, round(255 * x)) for x in dataset_conf["mean"]]),)
    print('===== auto_choice =====', dataset_conf["aug_choice"])
    print('===== magnitude =====', magnitude)
    train_transform = rand_augment_transform('rand-muni0-w0', aa_params, aug_choice=dataset_conf["aug_choice"], magnitude=magnitude)  

    common_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=dataset_conf["mean"], std=dataset_conf["std"])])
    train_common_transform = common_transform
    if dataset_conf["cutout_size"] > 0 and dataset_conf["name"]!='imagenet':
        train_common_transform.transforms.append(Cutout(dataset_conf["cutout_size"]))
    
    transform_list = [[train_transform, train_common_transform, None], # Train_transform  
                    [None, common_transform, None], # Train_transform_ori    
                    [train_transform, train_common_transform, None], # Val_transform  
                    [None, common_transform, None], # Val_transform_ori    
                    [None, common_transform, None]] # Test_transform 

    if dataset_conf["name"] in ['cifar10']:
        try:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_cifar10_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
        except: 
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_cifar10_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
    elif dataset_conf["name"] in ['cifar100']:
        try:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_cifar100_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
        except:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_cifar100_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
    
    elif dataset_conf["name"] in ['sport8','mit67','flowers102']:
        try:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_image224_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
        except:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_image224_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
    
    elif dataset_conf["name"] == 'imagenet':
        try:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_imagenet_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])
        except:
            train_data, train_data_ori, val_data, val_data_ori, test_data = get_imagenet_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, val_ratio=dataset_conf["val_ratio"], real_val=dataset_conf["real_val"])

    if not dataset_conf["random_sample"]: 
        # Without Replacement.   
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=dataset_conf["batch_size"], shuffle=True,
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
        train_loader_ori = torch.utils.data.DataLoader(
            train_data_ori, batch_size=dataset_conf["batch_size"], shuffle=True,
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
    
    elif dataset_conf["random_sample"]:
        # With Replacement.   
        num_train = len(train_data)
        indices = list(range(num_train))
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=dataset_conf["batch_size"], 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
        train_loader_ori = torch.utils.data.DataLoader(
            train_data_ori, batch_size=dataset_conf["batch_size"], 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])

    if dataset_conf["name"] != 'imagenet':
        val_loader=torch.utils.data.DataLoader(
            val_data, batch_size=dataset_conf["batch_size"], 
            sampler=FitSampler(train_data.__len__(), val_data.__len__()), 
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
        val_loader_ori=torch.utils.data.DataLoader(
            val_data_ori, batch_size=dataset_conf["batch_size"], 
            sampler=FitSampler(train_data.__len__(), val_data_ori.__len__()), 
            num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
    else:
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=dataset_conf["batch_size"]//2, 
            sampler=FitSampler(train_data.__len__()//2, val_data.__len__()),
            num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
        val_loader_ori = torch.utils.data.DataLoader(
            val_data_ori, batch_size=dataset_conf["batch_size"]//2, 
            sampler=FitSampler(train_data.__len__()//2, val_data_ori.__len__()),
            num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=dataset_conf["batch_size"], shuffle=False,
        num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])

    print(f"Length of training Loader: ------ {len(train_loader)}")
    print(f"Length of original training Loader: ------ {len(train_loader_ori)}")
    print(f"Length of validation Loader: ------ {len(val_loader)}")
    print(f"Length of validation Loader: ------ {len(val_loader_ori)}")
    print(f"Length of test Loader: ------ {len(test_loader)}")
    return train_loader, train_loader_ori, val_loader, val_loader_ori, test_loader
    
'''------------------------------------------Data Augmentation For Cifar10------------------------------------------'''
def build_dataset_darts_cifar(dataset_conf):    
    MEAN = dataset_conf['mean']
    STD = dataset_conf['std']
    transf = [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()]
    normalize = [transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)]
    train_transform = transforms.Compose(transf+normalize)
    if dataset_conf["cutout_size"] > 0:
        train_transform.transforms.append(Cutout(dataset_conf["cutout_size"]))
    valid_transform = transforms.Compose(normalize)
    transform_list = [train_transform, valid_transform, valid_transform]
    if dataset_conf['name'] == 'cifar10':
        try:
            train_data, val_data, test_data = get_cifar10_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, subsets=['train', 'val', 'test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
        except: 
            train_data, val_data, test_data = get_cifar10_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, subsets=['train', 'val', 'test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
    elif dataset_conf['name'] == 'cifar100':
        try:
            train_data, val_data, test_data = get_cifar100_dataset(dataset_conf["root"], 
                transform_list, dataset_conf, subsets=['train', 'val', 'test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
        except: 
            train_data, val_data, test_data = get_cifar100_dataset(dataset_conf["root_cloud"], 
                transform_list, dataset_conf, subsets=['train', 'val', 'test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
    print('The Length Of Training Data:', len(train_data))
    print('The Length Of Validation Data:', len(val_data))
    if not dataset_conf["random_sample"]:    
        # Without Replacement.   
        train_loader =  torch.utils.data.DataLoader(train_data, batch_size=dataset_conf['batch_size'], \
            shuffle=True, num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
        val_loader =  torch.utils.data.DataLoader(val_data, batch_size=dataset_conf["batch_size"], \
            sampler=FitSampler(train_data.__len__(), val_data.__len__()), num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=dataset_conf["batch_size"], \
            shuffle=False, num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    elif dataset_conf["random_sample"]:
        # With Replacement.   
        num_train = len(train_data)
        indices = list(range(num_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=dataset_conf['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=dataset_conf['batch_size']//2,
            sampler=FitSampler(train_data.__len__()//2, val_data.__len__()),
            num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin']) 
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=dataset_conf["batch_size"], 
            shuffle=False, num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    return train_loader, val_loader, test_loader

'''------------------------------------------Data Augmentation For Image224------------------------------------------'''
def build_dataset_darts_image224(dataset_conf):
    MEAN = dataset_conf['mean']
    STD = dataset_conf['std']
    normalize = transforms.Normalize(MEAN,STD)
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2),
      transforms.ToTensor(),
      normalize,])
    val_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,])
    if dataset_conf["cutout_size"] > 0:
        train_transform.transforms.append(Cutout(dataset_conf["cutout_size"]))
    transform_list = [train_transform, val_transform, val_transform]
    try:
        train_data, val_data_ori, test_data = get_image224_dataset(dataset_conf["root"], 
            transform_list, dataset_conf, subsets=['train','val_ori','test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
    except:
        train_data, val_data_ori, test_data = get_image224_dataset(dataset_conf["root_cloud"], 
            transform_list, dataset_conf, subsets=['train','val_ori','test'], val_ratio=dataset_conf["val_ratio"], choice='easy', real_val=dataset_conf["real_val"])
    train_loader =  torch.utils.data.DataLoader(train_data, batch_size=dataset_conf['batch_size'], shuffle=True, num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
    val_loader =  torch.utils.data.DataLoader(val_data_ori, batch_size=dataset_conf["batch_size"]//2, sampler=FitSampler(train_data.__len__()//2, val_data_ori.__len__()), num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=dataset_conf["batch_size"], shuffle=False, num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    return train_loader, val_loader, test_loader

'''------------------------------------------Data Augmentation For Imagenet------------------------------------------'''
def build_dataset_darts_imagenet(dataset_conf):    
    MEAN = dataset_conf['mean']
    STD = dataset_conf['std']
    normalize = transforms.Normalize(MEAN,STD)
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2),
      transforms.ToTensor(),
      normalize,])
    val_transform = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,])
    transform_list = [train_transform, val_transform, val_transform]
    try:
        train_data, val_data, test_data = get_imagenet_dataset(dataset_conf['root'], \
            transform_list, dataset_conf, subsets=['train', 'val', 'test'], \
            val_ratio=dataset_conf['val_ratio'], choice='easy', real_val=False)
    except:
        train_data, val_data, test_data = get_imagenet_dataset(dataset_conf['root_cloud'], \
            transform_list, dataset_conf, subsets=['train', 'val', 'test'], \
            val_ratio=dataset_conf['val_ratio'], choice='easy', real_val=False)
    train_loader =  torch.utils.data.DataLoader(train_data, batch_size=dataset_conf['batch_size'], shuffle=True, num_workers=dataset_conf['workers'], pin_memory=dataset_conf['pin'])
    val_loader =  torch.utils.data.DataLoader(val_data, batch_size=dataset_conf["batch_size"]//2, sampler=FitSampler(train_data.__len__()//2, val_data.__len__()), num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=dataset_conf["batch_size"], shuffle=False, num_workers=dataset_conf["workers"], pin_memory=dataset_conf['pin'])
    return train_loader, val_loader, test_loader

'''------------------------------------------Setting for Cifar10------------------------------------------'''
def get_cifar10_dataset(root, transform_list, dataset_conf, subsets=['train', 'train_ori', 'val', 'val_ori', 'test'], val_ratio=0, choice='hard', real_val=False):
    if isinstance(subsets, str):
        subsets = [subsets]
    out_datasets = []
    # Seperately applying it to 'train', 'val', 'test'.
    for subset, subtransform in zip(subsets, transform_list):
        if subset == 'test':
            data, labels = load_cifar_data(os.path.join(root, 'cifar-10-batches-py', 'test_batch'))
        else:
            npy_paths = [os.path.join(root, 'cifar-10-batches-py', 'data_batch_'+str(x_)) for x_ in range(1,6)]
            data, labels = load_cifar_data(npy_paths)
            train_size = int(len(labels)*(1-val_ratio))
            if subset in ['train','train_ori']:   
                if real_val:             
                    data = data[:train_size]
                    labels = labels[:train_size]
                elif not real_val:
                    data = data
                    labels = labels
            elif subset in ['val','val_ori']:
                data = data[train_size:]
                labels = labels[train_size:]
            else:
                assert False, "unknown subset type for cifar10 dataset"
        data = np.transpose(data.reshape(-1, 3, 32, 32), (0, 2, 3, 1)).astype(np.uint8)
        # choice = [hard, easy]
        if choice == 'hard':
            out_datasets.append(npy_dataset(data, labels, subtransform[0], subtransform[1], subtransform[2], dataset_conf))
        elif choice == 'easy':
            out_datasets.append(easy_dataset(data, labels, subtransform, dataset_conf))
    return (*out_datasets, )

def get_cifar100_dataset(root, transform_list, dataset_conf, subsets=['train', 'train_ori', 'val', 'val_ori', 'test'], val_ratio=0, choice='hard', real_val=False):
    if isinstance(subsets, str):
        subsets = [subsets]
    out_datasets = []
    data_train, labels_train = load_cifar_data(os.path.join(root, 'cifar-100-python', 'train'))
    data_test, labels_test = load_cifar_data(os.path.join(root, 'cifar-100-python', 'test'))
    train_size = int(len(labels_train) * (1 - val_ratio))
    for subset, subtransform in zip(subsets, transform_list):
        if subset in ['train','train_ori']:
            if real_val:
                data = data_train[:train_size]
                labels = labels_train[:train_size]
            elif not real_val:
                data = data_train
                labels = labels_train
        elif subset in ['val','val_ori']:
            data = data_train[train_size:]
            labels = labels_train[train_size:]
        elif subset == 'test':
            data, labels = data_test, labels_test
        else:
            assert False, "unknown subset type for cifar100 dataset"
        data = np.transpose(data.reshape(-1, 3, 32, 32), (0, 2, 3, 1)).astype(np.uint8)
        if choice == 'hard':
            out_datasets.append(npy_dataset(data, labels, subtransform[0], subtransform[1], subtransform[2], dataset_conf))
        elif choice == 'easy':
            out_datasets.append(easy_dataset(data, labels, subtransform, dataset_conf))
    return (*out_datasets, )

def load_cifar_data(file_path):
    if isinstance(file_path, list):
        data, full_labels = [], []
        for ipath in file_path:
            iData, iLabels = load_cifar_data(ipath)
            data.append(iData)
            full_labels.extend(iLabels)
        full_data = np.vstack(data)
        return full_data, full_labels
    else:    
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data = entry['data']
            if 'labels' in entry:
                labels = entry['labels']
            else:
                labels = entry['fine_labels']
    return data, labels

class npy_dataset(Dataset):
    # THis one is setting for multiple transforamtion applied to one set. 
    def __init__(self, np_array, labels, transform, second_transform, target_transform, dataset_conf):
        super(npy_dataset, self).__init__()
        self.data = np_array
        self.labels = labels
        self.transform = transform
        self.second_transform = second_transform
        # ToTensor + Normalize + Cutout in Second tranformation. 
        self.target_transform = target_transform
        self.flip = torchvision.transforms.RandomHorizontalFlip()
        self.dataset_conf = dataset_conf
        #self.flip = transforms.Compose(
        #    [transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip()])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.labels[idx]
        ops_code = np.zeros([self.dataset_conf["aug_choice"], 2])   
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            aug_img = self.flip(img)
            ensemble, ops_code = self.transform(aug_img, target)
            (aug_img, target) = ensemble
            aug_img = self.second_transform(aug_img)
        else:
            aug_img = self.second_transform(img)     
        return aug_img, np.array(target), ops_code

class easy_dataset(Dataset):
    def __init__(self, np_array, labels, transform, dataset_conf):
        super(easy_dataset, self).__init__()
        self.data = np_array
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.labels[idx]
        aug_img = self.transform(img)
        return aug_img, np.array(target)

'''------------------------------------------Batch Transformation For Image224------------------------------------------''' 
def get_image224_dataset(root, transform_list, dataset_conf, subsets=['train', 'train_ori', 'val', 'val_ori', 'test'], val_ratio=0, choice='hard', real_val=False):
    out_datasets = []
    # Seperately applying it to 'train', 'val', 'test'.
    for subset, subtransform in zip(subsets, transform_list):
        if subset == 'test':
            data_list = datasets.ImageFolder(root=os.path.join(root, 'test'))
            data_labels = []
            for i in data_list:
                data_labels.append(i)
        else:
            data_list = datasets.ImageFolder(root=os.path.join(root, 'train'))
            data_labels = []
            for i in data_list:
                data_labels.append(i)
            random.shuffle(data_labels)
            train_size = int(len(data_labels)*(1-val_ratio))
            if subset in ['train','train_ori']:   
                if real_val:          
                    data_labels = data_labels[:train_size]
                elif not real_val:
                    data_labels = data_labels
            elif subset in ['val','val_ori']:
                data_labels = data_labels[train_size:]
            else:
                assert False, "unknown subset type for cifar10 dataset"
        if choice == 'hard':
            out_datasets.append(path_dataset_loader(data_labels, subtransform[0], subtransform[1], subtransform[2], dataset_conf))
        elif choice == 'easy':
            out_datasets.append(easy_dataset_loader(data_labels, subtransform, dataset_conf))
    return (*out_datasets, )       

class path_dataset_loader(Dataset):
    def __init__(self, data_labels, transform, second_transform, target_transform, dataset_conf):
        super(path_dataset_loader, self).__init__()
        self.data_labels = data_labels
        self.transform = transform
        self.target_transform = target_transform
        self.second_transform = second_transform
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2)])
        self.valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224)])
        self.dataset_conf = dataset_conf
    def __len__(self):
        return len(self.data_labels)
    def __getitem__(self, idx):
        img, target = self.data_labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target) 
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.train_transform(img)
            ensemble, ops_code = self.transform(img, target)
            # The transformation considers both image and label to increase the possibility of transformation.
            (aug_img, target) = ensemble
            aug_img = self.second_transform(aug_img)
        elif self.second_transform is not None:
            img = self.valid_transform(img)
            aug_img = self.second_transform(img)
            ops_code = np.zeros([self.dataset_conf["aug_choice"], 2])
        return aug_img, target, ops_code

class easy_dataset_loader(Dataset):
    def __init__(self, data_labels, transform, dataset_conf):
        super(easy_dataset_loader, self).__init__()
        self.data_labels = data_labels
        self.transform = transform
    def __len__(self):
        return len(self.data_labels)
    def __getitem__(self, idx):
        img, target = self.data_labels[idx]
        img = img.convert('RGB')
        img = self.transform(img)
        return img, target 

'''------------------------------------------Batch Transformation For Imagenet------------------------------------------'''
def get_imagenet_dataset(root, transform_list, dataset_conf, subsets=['train', 'train_ori', 'val', 'val_ori', 'test'], val_ratio=0, choice='hard', real_val=False):
    if isinstance(subsets, str):
        subsets = [subsets]
    out_datasets = []
    data_train, data_val = load_imagenet_data(os.path.join(root, 'train'), val_ratio, real_val=False)
    data_test, _ = load_imagenet_data(os.path.join(root, 'val'), 0, real_val=False)
    for subset, subtransform in zip(subsets, transform_list):
        if subset == 'train':
            data_labels = data_train
        elif subset == 'train_ori':
            data_labels = data_train
        elif subset == 'val':
            data_labels = data_val
        elif subset == 'val_ori':
            data_labels = data_val
        elif subset == 'test':
            data_labels = data_test
        else:
            assert False, "unknown subset type for imagenet dataset"
        if choice == 'hard':
            out_datasets.append(path_dataset(data_labels, subtransform[0], subtransform[1], subtransform[2], dataset_conf))
        elif choice == 'easy':
            out_datasets.append(easy_path_dataset(data_labels, subtransform, dataset_conf))
    return (*out_datasets, )

def load_imagenet_data(file_path, val_ratio=0, real_val=False):
    if sys.version_info >= (3, 5):
        classes = [d.name for d in os.scandir(file_path) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    train_images, val_images = [], []
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(file_path, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            tmp_images = []
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if path.lower().endswith(extensions):
                    item = (path, class_to_idx[target])
                    tmp_images.append(item)
            train_size = int(len(tmp_images) * (1 - val_ratio))
            if real_val:
                train_images.extend(tmp_images[:train_size])
                val_images.extend(tmp_images[train_size:])
            elif not real_val:
                train_images.extend(tmp_images)
                val_images.extend(tmp_images[train_size:])
    return train_images, val_images

class path_dataset(Dataset):
    def __init__(self, data_labels, transform, second_transform, target_transform, dataset_conf):
        super(path_dataset, self).__init__()
        self.data_labels = data_labels
        self.transform = transform
        self.target_transform = target_transform
        self.second_transform = second_transform
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),])
        self.valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224)])
        self.dataset_conf = dataset_conf
    def __len__(self):
        return len(self.data_labels)
    def __getitem__(self, idx):
        path, target = self.data_labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target) 
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.train_transform(img)
            ensemble, ops_code = self.transform(img, target)
            # The transformation considers both image and label to increase the possibility of transformation.
            (aug_img, target) = ensemble
            aug_img = self.second_transform(aug_img)
        elif self.second_transform is not None:
            img = self.valid_transform(img)
            aug_img = self.second_transform(img)
            ops_code = np.zeros([self.dataset_conf["aug_choice"], 2])
        return aug_img, target, ops_code

class easy_path_dataset(Dataset):
    def __init__(self, data_labels, transform, dataset_conf):
        super(easy_path_dataset, self).__init__()
        self.data_labels = data_labels
        self.transform = transform
        self.cutout_size = dataset_conf["cutout_size"]
        if self.cutout_size is not None:
            self.cutout = Cutout(self.cutout_size)
    def __len__(self):
        return len(self.data_labels)
    def __getitem__(self, idx):
        path, target = self.data_labels[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform(img)
        if self.cutout_size is not None:
            img = self.cutout(img)
        return img, target

'''Supplementary Supporting'''
class Cutout(object): 
    def __init__(self, length):
        self.length = length
    def __call__(self, img1):
        h, w = img1.size(1), img1.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img1)
        img1 *= mask
        return img1

'''##########'''
class FitSampler(Sampler):
    def __init__(self, fit_len, data_len):
        self.num_samples = data_len
        self.out_len = fit_len
        self.remaid_index = []
    def __iter__(self):
        out_idx = self.remaid_index
        for _ in range((self.out_len-len(out_idx)-1)//self.num_samples+1):
            out_idx.extend(list(np.random.permutation(self.num_samples)))
        self.remaid_index = out_idx[self.out_len:]
        return iter(out_idx[:self.out_len])
    def __len__(self):
        return self.out_len

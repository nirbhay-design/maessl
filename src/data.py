import torch 
import torchvision 
import torchvision.transforms as transforms
import os, random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pickle 
from torch.utils.data.distributed import DistributedSampler

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

def get_transforms(image_size, data_name = "cifar10", algo='supcon'):
    if data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif data_name == "tinyimagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # for solarization 
    solarize_algo = ["vicreg", "bt"]
    solarize_p = 0.0
    if any([i in algo for i in solarize_algo]):
        solarize_p = 0.1
    
    s = 0.5

    # for smaller image datasets, no gaussian blur 
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        transforms.RandomGrayscale(p = 0.2),
        Solarization(p = 0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std=std)
    ])

    train_transforms_prime = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        transforms.RandomGrayscale(p = 0.2),
        Solarization(p = solarize_p),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std=std)
    ])

    train_transforms_mlp = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = mean, std = std)])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])
    if algo != "test":
        print(f"augmentation for {algo}: ")
        print(train_transforms)
        print(train_transforms_prime)

    return {"train_transforms": train_transforms, 
            "train_transforms_prime": train_transforms_prime, 
            "train_transforms_mlp": train_transforms_mlp, 
            "test_transforms": test_transforms}

class CustomImagenetTrainDataset():
    def __init__(self,img_path, wnids_path, n_class, pretrain=True, transform=None):
        self.img_path = img_path
        with open(wnids_path) as f:
            self.wnids = f.read().split('\n')
            self.wnids.remove('')
        self.wnids = sorted(self.wnids,key = lambda x:x)
        self.mapping = dict(list(zip(self.wnids,list(range(n_class)))))

        img_class = os.listdir(self.img_path)
        self.img_map = []
        for clss in img_class:
            cls_imgs = os.listdir(os.path.join(self.img_path,clss,'images'))
            clss_imgs = list(map(lambda x:[clss,x],cls_imgs))
            self.img_map.extend(clss_imgs)

        self.pretrain = pretrain
        if self.pretrain:
            self.target_transform = transform.get("train_transforms", None)
            self.target_transform_prime = transform.get("train_transforms_prime", None)
        
        else:
            self.transform_mlp = transform
            
    def __len__(self):
        return len(self.img_map)

    def __getitem__(self,idx):
        class_image,image_name = self.img_map[idx]
        cls_idx = self.mapping.get(class_image,-1)

        img = Image.open(os.path.join(self.img_path,class_image,'images',image_name)).convert('RGB')
        if self.pretrain:
            img1 = self.target_transform(img)
            img2 = self.target_transform_prime(img)
            return img1, img2, cls_idx 
        else:
            img = self.transform_mlp(img)
        return (img, cls_idx)
    
class CustomImagenetTestDataset():
    def __init__(self,img_path, wnids, test_anno, n_class, transform=None):
        self.img_path = img_path
        with open(wnids) as f:
            self.wnids = f.read().split('\n')
            self.wnids.remove('')

        with open(test_anno) as f:
            self.test_anno = list(map(lambda x:x.split('\t')[:2],f.read().split("\n")))
            self.test_anno.remove([''])

        self.wnids = sorted(self.wnids,key = lambda x:x)
        self.mapping = dict(list(zip(self.wnids,list(range(n_class)))))
        # self.rev_mapping = {j:i for i,j in self.mapping.items()}
        self.transformations = transform

    def __len__(self):
        return len(self.test_anno)

    def __getitem__(self,idx):
        test_img, class_name = self.test_anno[idx]
        cls_idx = self.mapping.get(class_name,-1)

        img = Image.open(os.path.join(self.img_path,test_img)).convert('RGB')
        img = self.transformations(img)
        return (img,cls_idx)

class DataCifar():
    def __init__(self, algo = "simclr", data_name = "cifar10", data_dir = "datasets/cifar10", target_transform = {}):
        if data_name == "cifar10":
            self.data = torchvision.datasets.CIFAR10(data_dir, train = True, download = True)
        elif data_name == "cifar100":
            self.data = torchvision.datasets.CIFAR100(data_dir, train = True, download = True)

        self.algo = algo
        self.target_transform = target_transform.get("train_transforms", None)
        self.target_transform_prime = target_transform.get("train_transforms_prime", None)
    
        if self.algo == "triplet":
            len_data = len(self.data)
            data_classes = len(self.data.classes)
            self.all_data = {i:[] for i in range(data_classes)}

            for idx in range(len_data):
                self.all_data[self.data[idx][1]].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.algo == "triplet":
            anc, anc_label = self.data[idx]

            pos_idx = random.choice(self.all_data[anc_label])

            all_classes_idx = list(self.all_data.keys())
            all_classes_idx.remove(anc_label)
            neg_label = random.choice(all_classes_idx)

            neg_idx = random.choice(self.all_data[neg_label])

            pos, pos_label = self.data[pos_idx]
            neg, neg_label = self.data[neg_idx]

            anc = self.target_transform(anc)
            pos = self.target_transform(pos)
            neg = self.target_transform(neg)

            return anc, anc_label, pos, pos_label, neg, neg_label
            
        image, label = self.data[idx]

        img1 = self.target_transform(image)
        img2 = self.target_transform_prime(image)
        return img1, img2, label 

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    all_transforms = get_transforms(image_size, data_name = "cifar100", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']


    train_transforms = {"train_transforms": all_transforms["train_transforms"],
                        "train_transforms_prime": all_transforms["train_transforms_prime"]}

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar100", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR100(
        data_dir,
        transform = all_transforms["train_transforms_mlp"],
        train = True,
        download = True
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform= all_transforms["test_transforms"],
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset

def Cifar10DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    all_transforms = get_transforms(image_size, data_name = "cifar10", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_transforms = {"train_transforms": all_transforms["train_transforms"],
                        "train_transforms_prime": all_transforms["train_transforms_prime"]}

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar10", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR10(
        data_dir,
        transform = all_transforms["train_transforms_mlp"],
        train = True,
        download = True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        data_dir, 
        transform= all_transforms["test_transforms"],
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset

def tinyimagenet_dataloader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    all_transforms = get_transforms(image_size, data_name = "tinyimagenet", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_transforms = {"train_transforms": all_transforms["train_transforms"],
                        "train_transforms_prime": all_transforms["train_transforms_prime"]}

    image_path = os.path.join(data_dir, "train")
    wnids_path = os.path.join(data_dir, "wnids.txt")
    test_image_path = os.path.join(data_dir, "val", "images")
    test_anno_path = os.path.join(data_dir, "val", "val_annotations.txt")
    n_class = 200

    train_dataset = CustomImagenetTrainDataset(img_path = image_path, wnids_path = wnids_path, 
                                               n_class = n_class, pretrain=True, transform = train_transforms)

    train_dataset_mlp = CustomImagenetTrainDataset(img_path = image_path, wnids_path = wnids_path, 
                                               n_class = n_class, pretrain=False, transform = all_transforms["train_transforms_mlp"])
    
    test_dataset = CustomImagenetTestDataset(img_path = test_image_path, wnids = wnids_path, test_anno = test_anno_path,
                                             n_class = n_class, transform = all_transforms["test_transforms"])

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset
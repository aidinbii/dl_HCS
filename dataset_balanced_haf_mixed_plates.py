import cv2
import logging
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#python_code = __import__('kaggle-rcic-1st')
#import python_code

# import precomputed as P
import ipdb


def tta(args, images):
    """Augment all images in a batch and return list of augmented batches"""

    ret = []
    n1 = math.ceil(args.tta ** 0.5)
    n2 = math.ceil(args.tta / n1)
    k = 0
    for i in range(n1):
        for j in range(n2):
            if k >= args.tta:
                break

            dw = round(args.tta_size * images.size(2))
            dh = round(args.tta_size * images.size(3))
            w = i * (images.size(2) - dw) // max(n1 - 1, 1)
            h = j * (images.size(3) - dh) // max(n2 - 1, 1)

            imgs = images[:, :, w:w + dw, h:h + dh]
            if k & 1:
                imgs = imgs.flip(3)
            if k & 2:
                imgs = imgs.flip(2)
            if k & 4:
                imgs = imgs.transpose(2, 3)

            ret.append(nn.functional.interpolate(
                imgs, images.size()[2:], mode='nearest'))
            k += 1

    return ret


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


def get_train_val_loader(args, predict=False):
    def train_transform1(image):
        #ipdb.set_trace()
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        if random.random() < 0.5:
            image = image[::-1, :, :]
        if random.random() < 0.5:
            image = image.transpose([1, 0, 2])
        image = np.ascontiguousarray(image)

        if args.scale_aug != 1:
            size_x = random.randint(
                round(1360 * args.scale_aug), 1360)  # change to 1024?
            size_y = random.randint(round(1024 * args.scale_aug), 1024)
            x = random.randint(0, 1360 - size_x)
            y = random.randint(0, 1024 - size_y)
            image = image[x:x + size_x, y:y + size_y]
            image = cv2.resize(image, (1360, 1024),
                               interpolation=cv2.INTER_NEAREST)

        return image
    
    def train_transform2(image):
        #ipdb.set_trace()
        # image.shape[0] = number of channels
        a, b = np.random.normal(1, args.pw_aug[0], (image.shape[0], 1, 1)), np.random.normal(
            0, args.pw_aug[1], (image.shape[0], 1, 1))
        a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(
            b, dtype=torch.float32)
        return image * a + b

    if not predict:
        train_dataset = CellularDataset(args.data, 'train_all_controls' if args.all_controls_train else 'train_controls', transform=(
            train_transform1, train_transform2), stain=args.stain, zone = args.zone, concat_zeroes=args.concat_zeroes, cv_number=args.cv_number, split_seed=args.data_split_seed, normalization=args.data_normalization)
        # ipdb.set_trace()
        # train = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_data_workers, worker_init_fn=worker_init_fn)

    for i in range(1 if not predict else 2):
        val_dataset = CellularDataset(args.data, 'val' if i == 0 else 'train', stain=args.stain,  concat_zeroes=args.concat_zeroes, zone = args.zone,
                                      cv_number=args.cv_number, split_seed=args.data_split_seed, normalization=args.data_normalization)
        # ipdb.set_trace()
        #assert(not set(train_dataset.data).isdisjoint(dataset.data)) == True
        #print (len(set(train_dataset.data).intersection(set(dataset.data))))
        # loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers, worker_init_fn=worker_init_fn)
        # if i == 0:
        #     val = loader
        # else:
        #     train = loader

    assert len(set(train_dataset.data).intersection(
        set(val_dataset.data))) == 0

    return train_dataset, val_dataset


def get_test_loader(args):
    test_dataset = CellularDataset(args.data, 'test', stain=args.stain, zone = args.zone, concat_zeroes=args.concat_zeroes,
                                   normalization=args.data_normalization)
    return DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers,
                      worker_init_fn=worker_init_fn)


class CellularDataset(Dataset):
    treatment_classes = 3

    def __init__(self, root_dir, mode, stain, zone, concat_zeroes, split_seed=1, cv_number=0, transform=None, normalization='global'):
        """
        :param split_seed: seed for train/val split of labeled experiments
        :param mode: possible choices:
                        train -- dataset containing only non-control images from training set
                        train_controls -- dataset containing non-control and control images from training set
                        train_all_controls -- dataset containing non-control and control images from training set and control images from validation and test set
                        val -- dataset containing only non-control images from validation set
                        test -- dataset containing only non-control images from test set
        :param transform: tuple of 2 functions for image transformation. First is called right after loading with image in numpy format. Second is called after normalization and converting to tensor
        """

        super().__init__()

        self.root = Path(root_dir)
        self.transform = transform
        self.concat_zeroes = concat_zeroes
        self.stain = stain

        assert normalization in ['global', 'experiment', 'sample']
        self.normalization = normalization

        if mode == 'train_controls':
            mode = 'train'
            move_controls = True
            all_controls = False
        elif mode == 'train_all_controls':
            mode = 'train'
            move_controls = True
            all_controls = True
        else:
            move_controls = False
            all_controls = False

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        
        if zone == 'low':
            self.csv = pd.read_csv(self.root / (f'low_stain{stain}.csv'))
        elif zone == 'high':
            self.csv = pd.read_csv(self.root / (f'high_stain{stain}.csv'))
                                   
        else:
            self.csv = pd.read_csv(
                self.root / (f'train_balanced_haf_mixed_plates.csv' if mode in ['train', 'val'] else f'test_balanced_haf_mixed_plates.csv'))
            
            '''
            self.csv = pd.read_csv(
                self.root / (f'train_downsampled_stain12_medium_atp.csv' if mode in ['train', 'val'] else f'test_downsampled_stain12_medium_atp.csv'))
            '''
       
       # row 	column 	Wellname 	plate 	drug 	concentration 	stain 	drug_encoded
        self.data = []

        for row in self.csv.iterrows():
            r = row[1]
            for field in range(2, 10):  # taking fields from 2 to 9!
                self.data.append((r.stain, r.plate, r.Wellname, r.row, r.column,
                                  field, r.drug, r.concentration, r.drug_type, r.drug_type_encoded))

        # randomly pick 10% of len(self.data) indices for validation
        #validation_indices = [random.randrange(0, len(self.data)) for i in range(0.1 * len(self.data) )]
        # print(foo[random_index])
        # ipdb.set_trace()
        if mode != 'test':
            # for reproducibilty of splitting
            state = random.Random(split_seed)
            if cv_number != -1:
                # ipdb.set_trace()
                val = state.sample(self.data, int(0.1 * len(self.data)))
            else:
                val = []

            #all = self.data.copy()
            #tr = sorted(set(all) - set(val))
            tr = [item for item in self.data if item not in val]
            #print (len(set(tr).intersection(set(val))))
            if mode == 'train':
                #logging.info('Train dataset: {}'.format(sorted(tr)))
                # ipdb.set_trace()
                #self.data = list(filter(lambda d: d in tr, self.data))
                self.data = tr
            elif mode == 'val':
                #logging.info('Val dataset: {}'.format(val))
                # ipdb.set_trace()
                #self.data = list(filter(lambda d: d in val, self.data))
                self.data = val

                #validat_set = random.sample(self.data, int(0.1 * len(self.data)))
                #train_set = [item for item in self.data if item not in validat_set]
            else:
                assert 0

        '''
        assert len(set(self.data)) == len(self.data)
        assert len(set(all_data)) == len(all_data)
            '''

        self.filter()

        logging.info('{} dataset size: data: {}'.format(mode, len(self.data)))

    def filter(self, func=None):
        """
        Filter dataset by given function. If function is not specified, it will clear current filter
        :param func: func((index, (experiment, plate, well, site, cell_type, sirna or None))) -> bool
        """
        if func is None:
            self.data_indices = None
        else:
            self.data_indices = list(
                filter(lambda i: func(i, self.data[i]), range(len(self.data))))

    def __len__(self):
        return len(self.data_indices if self.data_indices is not None else self.data)

    def __getitem__(self, i):
        i = self.data_indices[i] if self.data_indices is not None else i
        d = self.data[i]

       # i {Unnamed: 0-> Unnamed:1 	 row 	column 	Wellname 	plate 	drug 	concentration 	stain 	drug,concentration 	drug,concentration_encoded}
        #image_metadata = self.csv.iloc[i]
        # e.g. 014011-4-001001003: Row14Column11-ImageField4-Channel3
        # 002001-1-001001001
        # img_name = os.path.join(self.root,
        #                        self.landmarks_frame.iloc[idx, 0])

        #channels_stain1 = [1, 2, 3, 4]
        #channels_stain2 = [1, 2, 3]

       # d = {stain, plate, wellname, row, column, field, drug, concentration, drug_encoded}
        stain = d[0]
        plate = d[1]
        row = d[3]
        column = d[4]
        field = d[5]
        concentration = d[7]
        drug = d[6]
        #concentration = image_metadata[6]
        label = d[-1]

        # to account for single digit positions in the names
        if row < 10:
            row = '0' + str(row)
        if column < 10:
            column = '0' + str(column)

        d1 = self.root / f'Stain1Plate{plate}' / \
            f'stain1plate{plate}'
        d2 = self.root / f'Stain2Plate{plate}' / \
            f'stain2plate{plate}'
        imgs = []
        
        channels1 = [2]
        channels2 = [1, 2, 3]
        
        if self.stain == 1:
            channels1 = [1,2,3,4]
            channels2 = []
        if self.stain == 2:
            channels1 = []
            channels2 = [1,2,3]

        for channel in channels1:
            path = d1 / f'0{row}0{column}-{field}-00100100{channel}.tif'
            imgs.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        
        for channel in channels2:
            path = d2 / f'0{row}0{column}-{field}-00100100{channel}.tif'
            imgs.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))    

        image = np.stack(imgs, axis=-1)

        #image = np.array(image).transpose(1, 0, 2)

        # print (image.shape) # (1024, 1360, 3)
        # logging.info('shape{}'.format(image.shape))
        
        image = np.transpose(image, (1, 0, 2))  # (1360, 1024, 3)
        
        if self.transform is not None:
            image = self.transform[0](image)

        image = F.to_tensor(image)  # changes shape to (4, 1024, 1360)

        mean = torch.mean(image, dim=(1, 2))
        std = torch.std(image, dim=(1, 2))

        image = (image - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
        # logging.info('shape{}'.format(image.shape))

        if self.transform is not None:
            image = self.transform[1](image)

        #image = image.permute(0, 2, 1)

        # cell_type = nn.functional.one_hot(torch.tensor(self.cell_types.index(d[-2]), dtype=torch.long),
            # len(self.cell_types)).float()

        # add additional zero-value channels to make the image six channel. for transfer learning
        if self.concat_zeroes == True:
            if stain == 1:
                image = torch.cat(
                    (image, torch.zeros(2, image.shape[1], image.shape[2])), dim=0)
            else:
                image = torch.cat(
                    (image, torch.zeros(3, image.shape[1], image.shape[2])), dim=0)

        r = [image, torch.tensor(stain, dtype=torch.long), torch.tensor(
            field, dtype=torch.long), torch.tensor(i, dtype=torch.long)]

        # if self.mode != 'test':
        #r.append(torch.tensor(label, dtype=torch.long))

        r.append(torch.tensor(label, dtype=torch.long))
        # image, stain, field, index, label
        return tuple(r)

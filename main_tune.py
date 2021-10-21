#!/usr/bin/env python3

import itertools
import logging
import math
import pickle
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
#import pytorch_metric_learning.utils.logging_presets as logging_presets
import logging_presets_tune as logging_presets
import pytorch_metric_learning
from pytorch_metric_learning.utils import common_functions as c_f


#import dataset
#import dataset_haf
#import dataset_atp_exclud
#import dataset_stain12
#import dataset_upsampled_atp
#import dataset_upsampled_atp_haf
#import dataset_upsampled_atp_haf_split_plates

import importlib

from model import BackboneOnly, Backbone, Classifier
from train_with_classifier_hcs import TrainWithClassifierHCS

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



#import precomputed as P
#from model import ModelAndLoss
#from model import ArcMarginProduct
#from model_rcic import ModelAndLoss_rcic


def parse_args():
    def lr_type(x):
        x = x.split(',')
        return x[0], list(map(float, x[1:]))

    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', default='train',
                        choices=('train', 'val', 'predict'))
    
    parser.add_argument('--zone', type=str, default = None, choices=('low', 'high', 'medium'))
    parser.add_argument('--stain', type=int)
   # parser.add_argument('--stain_for_model', type=int)
    parser.add_argument('--lr_const', type=float, default = 1.5e-4)
    parser.add_argument('--metric_loss_weight', type=float)
    #parser.add_argument('--classifier_loss_weight', type=float)
    parser.add_argument('--log_path', help='path to log folder')
    parser.add_argument('--tensorboard_path', help='path to tensorboard folder')
    parser.add_argument('--model_save_path', help='path to saved model folder')
   
    #parser.add_argument('--checkpoint_dir', help='path to saved models for finetuning')
    
    parser.add_argument('--ds', type=str, help='choose dataset')
    
    parser.add_argument('--bn_mom', type=float, default=0.05)
    
    parser.add_argument('--backbone', default='resnet18',
                        help='backbone for the architecture. '
                        'Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets. '
                        'For DenseNets, add prefix "mem-" for memory efficient version')
    parser.add_argument('--head-hidden', type=lambda x: None if not x else list(map(int, x.split(','))),
                        help='hidden layers sizes in the head. Defaults to absence of hidden layers')
    parser.add_argument('--concat-cell-type', type=bool_type, default=True)
        
    parser.add_argument('--concat-zeroes', type=bool_type, default=False)
    
    parser.add_argument('--metric-loss-coeff', type=float, default=0.2)
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', '--ls', type=float, default=0)
    parser.add_argument('--mixup', type=float, default=0,
                        help='alpha parameter for mixup. 0 means no mixup')
    parser.add_argument('--cutmix', type=float, default=1,
                        help='parameter for beta distribution. 0 means no cutmix')

    parser.add_argument('--classes', type=int, default=13,
                        help='number of classes predicting by the network')
    parser.add_argument('--fp16', type=bool_type, default=True,
                        help='mixed precision training/inference')
    parser.add_argument('--disp-batches', type=int, default=50,
                        help='frequency (in iterations) of printing statistics of training / inference '
                        '(e.g. accuracy, loss, speed)')

    parser.add_argument('--tta', type=int,
                        help='number of TTAs. Flips, 90 degrees rotations and resized crops (for --tta-size != 1) are applied')
    parser.add_argument('--tta-size', type=float, default=1,
                        help='crop percentage for TTA')

    parser.add_argument('--save',
                        help='path for the checkpoint with best accuracy. '
                        'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
                        help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--pred-suffix', default='',
                        help='suffix for prediction output. '
                        'Predictions output will be stored in <loaded checkpoint path>.output<pred suffix>')

    parser.add_argument('--pw-aug', type=lambda x: tuple(map(float, x.split(','))), default=(0.1, 0.1),
                        help='pixel-wise augmentation in format (scale std, bias std). scale will be sampled from N(1, scale_std) '
                        'and bias from N(0, bias_std) for each channel independently')
    parser.add_argument('--scale-aug', type=float, default=0.5,
                        help='zoom augmentation. Scale will be sampled from uniform(scale, 1). '
                        'Scale is a scale for edge (preserving aspect)')
    parser.add_argument('--all-controls-train', type=bool_type, default=True,
                        help='train using all control images (also these from the test set)')
    parser.add_argument('--data-normalization', choices=('global', 'experiment', 'sample'), default='sample',
                        help='image normalization type: '
                        'global -- use statistics from entire dataset, '
                        'experiment -- use statistics from experiment, '
                        'sample -- use mean and std calculated on given example (after normalization)')
    parser.add_argument('--data', type=Path, default=Path('/storage/groups/peng/datasets/kenji/'),
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--cv-number', type=int, default=0, choices=(-1, 0, 1, 2, 3, 4, 5),
                        help='number of fold in 6-fold split. '
                        'For number of given cell type experiment in certain fold see dataset.py file. '
                        '-1 means not using validation set (training on all data)')
    parser.add_argument('--data-split-seed', type=int, default=0,
                        help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=6,
                        help='number of data loader workers')
    parser.add_argument('--seed', type=int,
                        help='global seed (for weight initialization, data sampling, etc.). '
                        'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('--pl-epoch', type=int, default=None,
                        help='first epoch where pseudo-labeling starts')
    parser.add_argument('--pl-size-func', type=str, default='x',
                        help='function indicating percentage of the test set transferred to the training set. '
                        'Function is called once an epoch and argument "x" is number from 0 to 1 indicating '
                        'training progress (0 is first epoch of pseudo-labeling, and 1 is last epoch of traning). '
                        'For example: "x" -- constant number of test examples is added each epoch; '
                        '"x*0.6+0.4" -- 40%% of test set added at the begining of pseudo-labeling and '
                        'then constant number each epoch')

    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=90)
    parser.add_argument('-l', '--lr', type=lr_type, default=('cosine', [1.5e-4]),
                        help='learning rate values and schedule given in format: schedule,value1,epoch1,value2,epoch2,...,value{n}. '
                        'in epoch range [0, epoch1) initial_lr=value1, in [epoch1, epoch2) initial_lr=value2, ..., '
                        'in [epoch{n-1}, total_epochs) initial_lr=value{n}, '
                        'in every range the same learning schedule is used. Possible schedules: cosine, const')
    args = parser.parse_args()
    
    '''
    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'val':
        assert args.save is None
    if args.mode == 'predict':
        assert args.load is not None
        assert args.save is None
    '''
    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args


def my_data_and_label_getter(output):
    '''A function that takes the output of your dataset's __getitem__ function, and 
     returns a tuple of (data, labels). 
     If None, then it is assumed that __getitem__ returns (data, labels).'''
    # output <-> (image, cell_type, index, sirna)
    X, labels = output[0], output[4]
    return X, labels



def subset_fields(res_list, num_fields = 8):
    res = []
    for i in range(0, len(res_list) - 1, num_fields):
        res.append([res_list[i], res_list[i + 1]])
    return res



def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.mode == 'train':
        handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    if args.mode == 'predict':
        handlers.append(logging.FileHandler(
            args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.DEBUG, format=head,
                        style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


@torch.no_grad()
def infer(args, model, loader, device):
    """Infer and return prediction in dictionary formatted {sample_id: logits}"""

    if not len(loader):
        return {}
    res = {}

    model.eval()
    tic = time.time()
    for i, (X, _, _, I, *_) in enumerate(loader):
        X = X.to(device)
        Xs = dataset.tta(args, X) if args.tta else [X]
        ys = [model.module.forward(X) for X in Xs]
        y = torch.stack(ys).mean(0).cpu()

        for j in range(len(I)):
            assert I[j].item() not in res
            res[I[j].item()] = y[j].numpy()

        if (i + 1) % args.disp_batches == 0:
            logging.info('Infer Iter: {:4d}  ->  speed: {:6.1f}'.format(
                i + 1, args.disp_batches * args.batch_size / (time.time() - tic)))
            tic = time.time()

    return res


def predict(args, model, device):
    """Entrypoint for predict mode"""

    test_loader = dataset.get_test_loader(args)
    train_loader, val_loader = dataset.get_train_val_loader(args, predict=True)

    # if args.fp16:
    # model = amp.initialize(model, opt_level='O1')

    logging.info('Starting prediction')

    output = {}
    for k, loader in [('test', test_loader),
                      ('val', val_loader)]:
        output[k] = {}

        # Enables autocasting for the forward pass (model + loss)
        with torch.cuda.amp.autocast():
            #res = infer(args, model, loader, device)
             output[k] = score(args, model, loader, device)
            
    return output


def score(args, model, loader, device):
    """Return accuracy of the model on validation set"""

    logging.info('Starting validation')

    res = infer(args, model, loader, device)

    # # number of examples for given cell type
    # cell_type_c = np.array([0, 0, 0, 0])
    # # number of correctly classified examples for given cell type
    # cell_type_s = np.array([0, 0, 0, 0])
    # for i, v in res.items():
    #     d = loader.dataset.data[i]
    #     r = v[:loader.dataset.treatment_classes].argmax() == d[-1]

    #     ser = loader.dataset.cell_types.index(d[4])
    #     cell_type_c[ser] += 1
    #     cell_type_s[ser] += r

    # acc = (cell_type_s.sum() / cell_type_c.sum()
    #        ).item() if cell_type_c.sum() != 0 else 0
    # logging.info('Eval: acc: {} ({})'.format(cell_type_s / cell_type_c, acc))
    n = 0
    s = 0
    for i, v in res.items():
        d = loader.dataset.data[i]
        r = v[:loader.dataset.treatment_classes].argmax() == d[-1]

        n += 1
        s += r

    acc = s / n if n != 0 else 0
    logging.info('Eval: acc: {}'.format(acc))

    return acc

def predict_test_labels(args, model, device, avg_fields = False):
    """Return dictionary of predicted labels of the model on test set and  ground truth and accuracy"""
    
    test_loader = dataset.get_test_loader(args)
    #train_loader, val_loader = dataset.get_train_val_loader(args, predict=True)
    
    with torch.cuda.amp.autocast():
        res = infer(args, model, test_loader, device)
    
    n = 0
    s = 0
    output = {}
    
    if avg_fields:
        # Predictions from different fields of a well are combined by taking mean of logits
        res_list = list(res.values())
        comb_logits = subset_fields(res_list, num_fields = 8)
        logits = np.average(comb_logits, axis = 1)
        pred = np.argmax(logits, axis = 1)
        
        test_loader.dataset.filter(lambda i, d: (i % 8 == 0)) # take one img from every well
        
        for i in range(len(pred)):
            output[i] = {}
            d = test_loader.dataset.data[i]
            output[i]['prediction'] = pred[i]
            output[i]['truth'] = d[-1]
        
            r = output[i]['prediction'] == d[-1]
            n += 1
            s += r
        
        output['acc'] = s / n if n != 0 else 0  
        return output

    for i, v in res.items():
        output[i] = {}
        d = test_loader.dataset.data[i]
        output[i]['prediction'] = v[:13].argmax() # is this correct?
        output[i]['truth'] = d[-1]
        
        r = output[i]['prediction'] == d[-1]
        n += 1
        s += r
        
    output['acc'] = s / n if n != 0 else 0    
    
    return output

    

def get_learning_rate(args, epoch):
    assert len(args.lr[1][1::2]) + 1 == len(args.lr[1][::2])
    for start, end, lr, next_lr in zip([0] + args.lr[1][1::2],
                                       args.lr[1][1::2] + [args.epochs],
                                       args.lr[1][::2],
                                       args.lr[1][2::2] + [0]):
        if start <= epoch < end:
            if args.lr[0] == 'cosine':
                return lr * (math.cos((epoch - start) / (end - start) * math.pi) + 1) / 2
            elif args.lr[0] == 'const':
                return lr
            else:
                assert 0
    assert 0


@torch.no_grad()
def smooth_label(args, Y):
    nY = nn.functional.one_hot(Y, args.classes).float()
    nY += args.label_smoothing / (args.classes - 1)
    nY[range(Y.size(0)), Y] -= args.label_smoothing / \
        (args.classes - 1) + args.label_smoothing
    return nY


@torch.no_grad()
def transform_input(args, X, Y):
    """Apply mixup, cutmix, and label-smoothing"""

    Y = smooth_label(args, Y)

    if args.mixup != 0 or args.cutmix != 0:
        perm = torch.randperm(args.batch_size).cuda()

    if args.mixup != 0:
        coeffs = torch.tensor(np.random.beta(
            args.mixup, args.mixup, args.batch_size), dtype=torch.float32).cuda()
        X = coeffs.view(-1, 1, 1, 1) * X + \
            (1 - coeffs.view(-1, 1, 1, 1)) * X[perm, ]
        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm, ]

    if args.cutmix != 0:
        img_height, img_width = X.size()[2:]
        lambd = np.random.beta(args.cutmix, args.cutmix)
        column = np.random.uniform(0, img_width)
        row = np.random.uniform(0, img_height)
        height = (1 - lambd) ** 0.5 * img_height
        width = (1 - lambd) ** 0.5 * img_width
        r1 = round(max(0, row - height / 2))
        r2 = round(min(img_height, row + height / 2))
        c1 = round(max(0, column - width / 2))
        c2 = round(min(img_width, column + width / 2))
        if r1 < r2 and c1 < c2:
            X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]

            lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)
            Y = Y * lambd + Y[perm] * (1 - lambd)

    return X, Y


def train_tune(config, checkpoint_dir=None):
    trunk = BackboneOnly(backbone=args.backbone, embedding_size=args.embedding_size, stain = args.stain, 
                         bn_mom = args.bn_mom) # try tweaking bn_mom

    '''
    embedder = Embedder(num_features=trunk.features_num,
                        embedding_size=embedding_size,
                        concat_cell_type=concat_cell_type,
                        bn_mom=bn_mom)
    '''
    classifier = Classifier(embedding_size=args.embedding_size,
                            classes=args.classes,
                            head_hidden=args.head_hidden,
                            bn_mom = args.bn_mom)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        trunk = nn.DataParallel(trunk)
        #embedder = nn.DataParallel(embedder)
        classifier = nn.DataParallel(classifier)

    trunk.to(device)
    #embedder.to(device)
    classifier.to(device)


    # Set optimizers
    #weight_decay = 1e-3
    #lr = 1.5e-4

    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=config["lr"], weight_decay=args.wd)
    #embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr, weight_decay=weight_decay)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], weight_decay=args.wd)

           
    # actually returns two CellularDataset types not dataloaders. Already augmented with transformations.
    dataset = importlib.import_module(args.ds)
    train_dataset, val_dataset = dataset.get_train_val_loader(args) # add option for dataset in args!

    # Set the metric loss function
    loss_func = losses.SupConLoss(temperature=0.1) # tune temperature!
    #loss_func = losses.ArcFaceLoss(num_classes = args.classes, embedding_size = args.embedding_size)
    #loss_func.to(device)
    #loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
    #loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-5) 

    #loss_func = losses.CircleLoss() # make an option for loss f. in args
    '''
    loss_func = losses.TripletMarginLoss(margin=0.05,
                                        swap=False,
                                        smooth_loss=False,
                                        triplets_per_anchor='all')
    '''
    # Set the classification loss:
    '''
    num_points_drug = train_dataset.csv.groupby('drug_encoded').count().iloc[:, 0].tolist()
    print ("number of points per drug", num_points_drug)
    class_weights = torch.FloatTensor(np.sum(num_points_drug) / num_points_drug).to(device)
    print ("class weights", class_weights)
    classification_loss = torch.nn.CrossEntropyLoss(weight = class_weights)
    '''
    classification_loss = torch.nn.CrossEntropyLoss()
    # Set the mining function
    #miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    #sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "classifier": classifier}

    optimizers = {"trunk_optimizer": trunk_optimizer, "classifier_optimizer": classifier_optimizer}
    #optimizers = {"trunk_optimizer": trunk_optimizer, "classifier_optimizer": classifier_optimizer, "metric_loss_optimizer": loss_optimizer}

    loss_funcs = {"metric_loss": loss_func, "classifier_loss": classification_loss}
    mining_funcs = {}
    
    metric_loss_weight = args.metric_loss_weight
    classifier_loss_weight = 1.0 - metric_loss_weight
    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": metric_loss_weight, "classifier_loss": classifier_loss_weight}


    accuracy_calculator = AccuracyCalculator(include=(),
                        exclude=(),
                        avg_of_avgs=False,
                        k=None,
                        label_comparison_fn=None)

    record_keeper, _, _ = logging_presets.get_record_keeper(args.log_path, args.tensorboard_path)
    hooks = logging_presets.get_hook_container(record_keeper, primary_metric="precision_at_1", save_models=True) #hooks is a HookContainer object
    dataset_dict = {"train": train_dataset, "val": val_dataset}
    model_folder = args.model_save_path
    
    
    #c_f.LOGGER.info("TRAINING EPOCH %d" % self.epoch)
    c_f.LOGGER.info('model folder{}'.format(model_folder))
    c_f.LOGGER.info('current lr constant', config["lr"])
    c_f.LOGGER.info('current batch size', config["batch_size"])
    c_f.LOGGER.info('len of train dataset', train_dataset.__len__())
    print('model folder', model_folder)
    '''
    def end_of_testing_hook(tester):
        print(tester.all_accuracies)
    '''

    #splits_to_eval = [('val', ['val', 'train'])]

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        normalize_embeddings=True, batch_size=config["batch_size"],
        dataloader_num_workers=args.num_data_workers,
        data_and_label_getter = my_data_and_label_getter,
        accuracy_calculator=accuracy_calculator,
        end_of_testing_hook=hooks.end_of_testing_hook)

    # Or if your model is composed of a trunk + embedder
    #all_accuracies = tester.test(dataset_dict, epoch, trunk, embedder)

    #end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, splits_to_eval = splits_to_eval)
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder) # wo splits_to_eval;make an option
    
    '''
    T_0 = 90
    eta_min = 1e-5
    verbose = True
    trunk_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trunk_optimizer,
                                            T_max = T_0,
                                            eta_min=eta_min,
                                            verbose=verbose)
    
    classifier_lr = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer,
                                            T_max = T_0,
                                            eta_min=eta_min,
                                            verbose=verbose)
    
    
    lr_schedulers = {"trunk_scheduler_by_epoch": trunk_lr,
                    "classifier_scheduler_by_epoch": classifier_lr}
    '''
    lr_schedulers = None
    if args.mode in ['train', 'val']:
        trainer = TrainWithClassifierHCS(models=models,
                                            optimizers=optimizers,
                                            loss_funcs=loss_funcs,
                                            mining_funcs=mining_funcs,
                                            batch_size=config["batch_size"],
                                            dataset = train_dataset,
                                            dataloader_num_workers = args.num_data_workers,
                                            data_and_label_getter = my_data_and_label_getter,
                                            loss_weights = loss_weights,
                                            lr_schedulers=lr_schedulers,
                                            end_of_iteration_hook = hooks.end_of_iteration_hook,
                                            end_of_epoch_hook=end_of_epoch_hook)

        trainer.train(num_epochs=args.epochs)
    
    elif args.mode == 'predict':
        predict(args, model)
    else:
        assert 0


def main(args):
    # parameters
    # Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets
    #backbone = 'resnet18'
    #embedding_size = 512
    #concat_cell_type = True
    #classes = 13
    # hidden layers sizes in the head. Defaults to absence of hidden layers
    #head_hidden = None
    #bn_mom = 0.05
    
    ray.init(include_dashboard=False) 
    config = {
        "lr": tune.loguniform(1e-7, 1e-3),
        "batch_size": tune.choice([24, 32, 64])
    }
    gpus_per_trial = torch.cuda.device_count()
    num_samples = 10
    max_num_epochs = 90
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["accuracy", "epoch", "training_iteration"])
    result = tune.run(
        train_tune,
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    

if __name__ == '__main__':
    args = parse_args()
    #setup_logging(args)
    setup_determinism(args)
    main(args)

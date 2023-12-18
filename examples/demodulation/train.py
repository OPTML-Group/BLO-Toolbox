import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
import os
import pickle
# from torch import dataloader
from torch.utils.data import DataLoader

# from torchmeta.datasets import Omniglot
# from torchmeta.utils.data import BatchMetaDataLoader
# from torchmeta.transforms import Categorical, ClassSplitter

from gbml.maml import MAML
from gbml.imaml import iMAML
from gbml.neumann import Neumann
from gbml.reptile import Reptile
from gbml.cavia import CAVIA
from gbml.fomaml import FOMAML
from gbml.signmaml import SignMAML
from utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker
import time

def train(args, model, dataloader):

    loss_list = []
    acc_list = []
    grad_list = []
    with tqdm(dataloader, total=args.num_train_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # import IPython; IPython.embed()
            loss_log, acc_log, grad_log = model.outer_loop(batch, is_train=True)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            grad_list.append(grad_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f} || grad={:.4f}'.format(np.mean(loss_list), np.mean(acc_list), np.mean(grad_list)))
            if batch_idx >= args.num_train_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, acc, grad

@torch.no_grad()
def valid(args, model, dataloader):

    loss_list = []
    acc_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)

    return loss, acc

@BestTracker
def run_epoch(epoch, args, model, train_loader, valid_loader):

    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_acc, train_grad = train(args, model, train_loader)
    valid_loss, valid_acc = valid(args, model, valid_loader)
    # test_loss, test_acc = valid(args, model, test_loader)

    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_acc'] = train_acc
    res['train_grad'] = train_grad
    res['valid_loss'] = valid_loss
    res['valid_acc'] = valid_acc
    res['test_loss'] = valid_loss
    res['test_acc'] = valid_acc
    
    return res

def main(args):

    if args.alg=='MAML':
        model = MAML(args)
    elif args.alg=='Reptile':
        model = Reptile(args)
    elif args.alg=='Neumann':
        model = Neumann(args)
    elif args.alg=='CAVIA':
        model = CAVIA(args)
    elif args.alg=='iMAML':
        model = iMAML(args)
    elif args.alg=='FOMAML':
        model = FOMAML(args)
    elif args.alg=='SignMAML':
        model = SignMAML(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.load:
        model.load()
    elif args.load_encoder:
        model.load_encoder()

    # load train data from pickle file
    with open(os.path.join(args.data_path, 'meta_train_3200.pckl'), 'rb') as f:
        train_dataset = pickle.load(f)
    # load test data from pickle file
    with open(os.path.join(args.data_path, 'meta_test_11600.pckl'), 'rb') as f:
        test_dataset = pickle.load(f)

    # import IPython; IPython.embed()
    test_dataset  = test_dataset[0]

    # import IPython; IPython.embed()
    # TODO: parse the train_dataset and test_dataset
    train_train_X = train_dataset[:,:args.num_shot,-2:]
    train_train_y = train_dataset[:,:args.num_shot,0:2]
    
    # repeat train_X for 996 times to be a pytorch tensor
    # train_train_X = train_train_X.repeat(50,1,1)
    # repeat train_y for 996 times to be a pytorch tensor
    # train_train_y = train_train_y.repeat(50,1,1)

    # train_test_X = torch.transpose(train_dataset[:,1:51,-2], 0, 1).reshape(-1,1)
    # train_test_y = torch.transpose(train_dataset[:,1:51,0], 0, 1).reshape(-1,1)

    train_test_X = train_dataset[:,args.num_shot:,-2:]
    train_test_y = train_dataset[:,args.num_shot:,0:2]
    
    # test_train_X = test_dataset[:,:1,-2].repeat(1000,1)
    # test_train_y = test_dataset[:,:1,0].repeat(1000,1)
    # test_test_X = test_dataset[:,1:1001,-2].reshape(-1,1)
    # test_test_y = test_dataset[:,1:1001,0].reshape(-1,1)

    test_train_X = test_dataset[:,:args.num_shot,-2:]
    test_train_y = test_dataset[:,:args.num_shot,0:2]
    test_test_X = test_dataset[:,args.num_shot:,-2:]
    test_test_y = test_dataset[:,args.num_shot:,0:2]


    # import IPython; IPython.embed()

    def map_label(x):
        return int((x+3)/2)

    # map all y's values from {-3.0,-1.0,1.0,3.0} to {0,1,2,3} and keep the type as torch tensor with the same shape as before, using lambda function
    # import IPython; IPython.embed()
    train_train_y = train_train_y.apply_(map_label)
    train_test_y = train_test_y.apply_(map_label)
    test_train_y = test_train_y.apply_(map_label)
    test_test_y = test_test_y.apply_(map_label)

    train_train_y = (train_train_y[:,:,0] * 4 + train_train_y[:,:,1]).long()
    train_test_y = (train_test_y[:,:,0] * 4 + train_test_y[:,:,1]).long()
    test_train_y = (test_train_y[:,:,0] * 4 + test_train_y[:,:,1]).long()
    test_test_y = (test_test_y[:,:,0] * 4 + test_test_y[:,:,1]).long()

    # merge train_train_X, train_train_y, train_test_X, train_test_y into a single dataset
    # for each batch of this dataset, we can load them as 
    # train_inputs, train_targets = batch['train']
    # test_inputs, test_targets = batch['test']
    
    train_data = []
    for i in range(len(train_train_X)):
        train_data.append({'train':(train_train_X[i], train_train_y[i]), 'test':(train_test_X[i], train_test_y[i])})
    test_data = []
    for i in range(len(test_train_X)):
        test_data.append({'train':(test_train_X[i], test_train_y[i]), 'test':(test_test_X[i], test_test_y[i])})
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)

    # import IPython; IPython.embed()

    # train_dataset = Omniglot(args.data_path, num_classes_per_task=args.num_way,
    #                     meta_split='train', 
    #                     transform=transforms.Compose([
    #                     transforms.RandomCrop(105, padding=8),
    #                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor(),
    #                     # transforms.Normalize(
    #                     #     np.array([0.485, 0.456, 0.406]),
    #                     #     np.array([0.229, 0.224, 0.225])),
    #                     ]),
    #                     target_transform=Categorical(num_classes=args.num_way),
    #                     download=True
    #                     )
    # train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    # train_loader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size,
    #     shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # valid_dataset = Omniglot(args.data_path, num_classes_per_task=args.num_way,
    #                     meta_split='val', 
    #                     transform=transforms.Compose([
    #                     transforms.CenterCrop(105),
    #                     transforms.ToTensor(),
    #                     # transforms.Normalize(
    #                     #     np.array([0.485, 0.456, 0.406]),
    #                     #     np.array([0.229, 0.224, 0.225]))
    #                     ]),
    #                     target_transform=Categorical(num_classes=args.num_way)
    #                     )
    # valid_dataset = ClassSplitter(valid_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    # valid_loader = BatchMetaDataLoader(valid_dataset, batch_size=args.batch_size,
    #     shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # test_dataset = Omniglot(args.data_path, num_classes_per_task=args.num_way,
    #                     meta_split='test', 
    #                     transform=transforms.Compose([
    #                     transforms.CenterCrop(105),
    #                     transforms.ToTensor(),
    #                     # transforms.Normalize(
    #                     #     np.array([0.485, 0.456, 0.406]),
    #                     #     np.array([0.229, 0.224, 0.225]))
    #                     ]),
    #                     target_transform=Categorical(num_classes=args.num_way)
    #                     )
    # test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    # test_loader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size,
    #     shuffle=True, pin_memory=True, num_workers=args.num_workers)
    start_time = time.time()
    for epoch in range(args.num_epoch):

        res, is_best = run_epoch(epoch, args, model, train_loader, valid_loader)
        res['time_elapsed'] = time.time() - start_time
        dict2tsv(res, os.path.join(args.result_path, args.alg, str(args.num_shot), args.log_path))

        if is_best:
            model.save()
            print('Best test accuracy is %.4f' % res['test_acc'])
        torch.cuda.empty_cache()

        if args.lr_sched:
            model.lr_sched()

    return None

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    # experimental settings
    parser.add_argument('--seed', type=int, default=2020,
        help='Random seed.')   
    parser.add_argument('--data_path', type=str, default='./dataset_prepare')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    # training settings
    parser.add_argument('--num_epoch', type=int, default=50,
        help='Number of epochs for meta train.') 
    parser.add_argument('--batch_size', type=int, default=1,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=1000,
        help='Number of batches the model is trained over (default: 250).')
    parser.add_argument('--num_valid_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=1,
        help='Number of support examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_query', type=int, default=15,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--alg', type=str, default='MAML')
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # network settings
    parser.add_argument('--net', type=str, default='MLP')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    main(args)
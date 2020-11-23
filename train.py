import torch
import torchvision
from utils.get_transforms import get_train_transform, get_test_transform, Target_Transformer
from VRPDataset import VRPDataset
from networks.TCNet import TCNet
import torch.optim as optim
import torch.nn as nn
from settings import params, fix_settings
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils import data
import os
import pickle
from torch.nn import MultiLabelSoftMarginLoss
from utils.hosvd import hosvd_dcmp
import numpy as np
import sys

def train(dataloader, net, criterion, optimizer, device):

    for epoch in range(params['max_epoches']):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs.view((outputs.shape[0], -1)), labels.view((labels.shape[0], -1)))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % params['print_freq'] == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss))
                running_loss = 0.0

        if (epoch+1)%params['lr_freq'] == 0:
            adjust_learning_rate(optimizer, epoch+1, params['lr'], params['lr_decay'], params['lr_freq'])

    print('Finished Training')


class TagTogLoss(nn.Module):

    def __init__(self, weight=None):
        super(TagTogLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        if self.weight is not None:
            target = target*self.weight
        t = torch.log(torch.softmax(torch.clamp(output, min=1e-30, max=10), dim=1)) * target/(torch.sum(target, dim=1, keepdim=True)+1e-30)
        return -torch.mean(torch.sum(t, dim=1))
        
def adjust_learning_rate(optimizer, epoch, lr_init, lr_decay, lr_freq):
    lr = lr_init * ( lr_decay ** (epoch // lr_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main(params):
    # reproducitiblity
    torch.manual_seed(0) 
    np.random.seed(0) 
    torch.backends.cudnn.deterministic = False # cuDNN deterministically select an algorithm, possibly at the cost of reduced performance

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('using device : ', device)

    # prepare dataset
    train_transform = get_train_transform()
    train_dataset = VRPDataset(params['dataset'], 'train', transform=train_transform)
    target_transformer = Target_Transformer()
    train_dataset.set_target_transformer(target_transformer)
    if params['weighted_loss'] == 'weighted_sample':
        sample_weights = pickle.load(open(os.path.join('data', params['dataset'], 'sample_weights.pkl'), 'rb'))
        weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, sampler=weighted_sampler, num_workers=8)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=8)

    train_global_tensor = pickle.load(open(os.path.join('data', params['dataset'], 'train_global_tensor.pkl'), 'rb'))
    U, S = hosvd_dcmp(train_global_tensor, params['epsilon'])
    pre_rank = np.array([train_dataset.num_entity, train_dataset.num_entity, train_dataset.num_predicate])
    new_rank = (U[0].shape[1], U[1].shape[1], U[2].shape[1])
    if params['model_name'] == 'TCN':
        net = TCNet(pre_rank, new_rank, U, params)
    else:
        print('unrecognized model name {}'.format(args.model))
    net = net.to(device)

    # define loss function and optimizer
    if params['weighted_loss'] == 'weighted_label':
        label_weights = pickle.load(open(os.path.join('data', params['dataset'], 'label_weights.pkl'), 'rb'))
        label_weights = torch.from_numpy(label_weights.flatten()).to(device)
        criterion = TagTogLoss(label_weights)
    else:
        criterion = TagTogLoss()

    model_params = list(net.parameters())


    optimizer = optim.SGD(model_params, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    # train
    train(train_loader, net, criterion, optimizer, device)

    # save model
    save_model_path = os.path.join('models', params['model_file_name'])
    torch.save(net.state_dict(), save_model_path)


if __name__ == '__main__':
    from settings import params, fix_settings
    fix_settings()
    main(params)

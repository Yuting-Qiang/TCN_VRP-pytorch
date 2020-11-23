import torch
import torchvision
from utils.get_transforms import get_test_transform
from VRPDataset import VRPDataset
from networks.TCNet import TCNet
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import pickle
import os
from utils.hosvd import hosvd_dcmp
import numpy as np

def test(dataloader, net, maxk, device):
    net.eval()
    # return top@maxk relationships for each image
    with torch.no_grad():
        predictions = []
        targets = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels[0]
            outputs = net(inputs)
            outputs = outputs[0]
            topk_index = np.transpose(
                    np.stack(
                        np.unravel_index(
                            np.argsort(outputs.cpu().numpy(), axis=None)[::-1][:maxk], shape=outputs.shape
                            )
                        )
                    )
            topk_values = outputs.cpu().numpy()[topk_index[:, 0], topk_index[:, 1], topk_index[:, 2]][:, None]
            predictions.append(np.hstack((topk_index, topk_values)))
            targets.append(np.stack(np.where(labels>0)).transpose())
    return predictions, targets

def main(params):
    # reproducitiblity
    torch.manual_seed(0) 
    np.random.seed(0) 
    torch.backends.cudnn.deterministic = False # cuDNN deterministically select an algorithm, possibly at the cost of reduced performance

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device : ', device)

    # prepare dataset
    test_transform = get_test_transform()
    test_dataset = VRPDataset(params['dataset'], 'test', transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    train_global_tensor = pickle.load(open(os.path.join('data', params['dataset'], 'train_global_tensor.pkl'), 'rb'))
    U, S = hosvd_dcmp(train_global_tensor, params['epsilon'])
    pre_rank = train_global_tensor.shape
    new_rank = (U[0].shape[1], U[1].shape[1], U[2].shape[1])

    # prepare network and load saved model
    if params['model_name'] == 'TCN':
        net = TCNet(pre_rank, new_rank, U, params)
    else:
        print('Unrecognized model name {}'.format(params['model_name']))
    net = net.to(device)

    save_model_path = os.path.join('models', params['model_file_name'])
    net.load_state_dict(torch.load(save_model_path, map_location=torch.device('cpu')))

    # train
    pred, target = test(test_loader, net, 100, device)
    np.save(os.path.join('results', params['dataset'], 'predictions.npy'), pred)
    np.save(os.path.join('results', params['dataset'], 'targets.npy'), target)

if __name__ == '__main__':
    from settings import params, fix_settings
    fix_settings()
    main(params)

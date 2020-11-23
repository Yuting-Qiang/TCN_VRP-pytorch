import os
import numpy as np
import torch
from PIL import Image
import pickle
from torch.utils import data

class VRPDataset(data.Dataset):
    def __init__(self, dataset, set, transform, target_transform=None):
        self._name = dataset
        annotations = pickle.load(open(os.path.join('data', self._name, 'annotations.pkl'), 'rb'))
        self.ind_to_entity = annotations['objects']
        self.entity_to_ind = {value:key for key, value in enumerate(self.ind_to_entity)}
        for key in self.entity_to_ind.keys():
            assert(self.ind_to_entity[self.entity_to_ind[key]] == key)
        self.ind_to_predicate = annotations['predicates']
        self.predicate_to_ind = {value:key for key, value in enumerate(self.ind_to_predicate)}
        for key in self.predicate_to_ind.keys():
            assert(self.ind_to_predicate[self.predicate_to_ind[key]] == key)
        if set == 'train': # remove those with no relations
            annotations = annotations['train_annotations']
        elif set == 'test':
            annotations = annotations['test_annotations']
        self.im_names = [x['img'] for x in annotations]
        self.rlp_labels = [x['relations'] for x in annotations]
        self.num_rlp_instance = 0
        for relations in self.rlp_labels:
            self.num_rlp_instance += len(relations)

        self.transform = transform
        self.target_transform = target_transform
        print('-- Performing Experiments on {} --'.format(self._name))
        print('Number of different entities : ', len(self.ind_to_entity))
        # for i in np.arange(self.num_entity):
            # print(' ({}: {}) '.format(i, self.ind_to_entity[i]))
        # print('Number of different predicates : ', len(self.ind_to_predicate))
        # for i in np.arange(self.num_predicate):
        #     print('({}: {})'.format(i, self.ind_to_predicate[i]))
        print('{} dataset loaded, {} images used, {} relationship instance in total'.format(set, len(self.im_names), self.num_rlp_instance))

    def __getitem__(self, index):
        img = Image.open(os.path.join('data', self._name, 'images', self.im_names[index].decode('utf-8') if self._name == 'vg' else self.im_names[index])).convert('RGB')
        # if img.ndim == 2:
        #     img = img[:, :, None][:, :, [0, 0, 0]]
        target = np.zeros((self.num_entity, self.num_entity, self.num_predicate), dtype=np.float32)
        classes = np.zeros(self.num_entity)
        for x in self.rlp_labels[index]:
            target[x[0], x[2], x[1]] += 1
            classes[x[0]] = 1
            classes[x[2]] = 1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get_im_name(self, index):
        return self.im_names[index]

    def __len__(self):
        return len(self.im_names)

    @property
    def num_entity(self):
        return len(self.ind_to_entity)

    @property
    def num_predicate(self):
        return len(self.ind_to_predicate)

    def set_target_transformer(self, target_transfromer=None):
        self.target_transform = target_transfromer


if __name__ == '__main__':
    dataset = 'vgvtranse'
    batch_size = 16
    from utils import get_train_transform, get_test_transform, construct_global_tensor, Target_Transformer
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_dataset = VRPDataset(dataset, 'train', transform=train_transform)
    test_dataset = VRPDataset(dataset, 'test', transform=test_transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    target_transformer = Target_Transformer()
    train_dataset.set_target_transformer(target_transfromer=target_transformer)
    test_dataset.set_target_transformer(target_transfromer=target_transformer)

    # network
    pre_rank = np.array([train_dataset.num_entity, train_dataset.num_entity, train_dataset.num_predicate])
    if not os.path.exists(os.path.join('data', dataset, 'cache')):
        os.mkdir(os.path.join('data', dataset, 'cache'))
    train_global_tensor_file = os.path.join('data', dataset, 'cache', 'train_global_tensor_unique.pkl')
    test_global_tensor_file = os.path.join('data', dataset, 'cache', 'test_global_tensor_unique.pkl')
    if os.path.exists(train_global_tensor_file):
        train_global_tensor = pickle.load(open(train_global_tensor_file, 'rb'))
    else:
        train_global_tensor = construct_global_tensor(train_loader, pre_rank)
        pickle.dump(train_global_tensor, open(train_global_tensor_file, 'wb'))
    if os.path.exists(test_global_tensor_file):
        test_global_tensor = pickle.load(open(test_global_tensor_file, 'rb'))
    else:
        test_global_tensor = construct_global_tensor(test_loader, pre_rank)
        pickle.dump(test_global_tensor, open(test_global_tensor_file, 'wb'))


# class XXXDataset(object):

    # def __init__(self, root, transforms):
        # self.root = root
        # self.transforms = transforms

    # def __getitem__(self, idx):
        # return image, target

    # def __len__(self):
        # return len(self.imgs)


# test XXDataset
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
if __name__ == '__main__':

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(np.img, (1,2,0)))
        plt.show()

    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    dataset = XXXDataset(root='./data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(labels)

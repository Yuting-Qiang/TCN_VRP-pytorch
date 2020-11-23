import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np

class TCNet(nn.Module):

    def __init__(self, pre_rank, new_rank, U, params, mask=None):
        super(TCNet, self).__init__()
        self.shareAB = params['shareAB']

        vgg = models.vgg16(pretrained=True)
        modules1 = list(vgg.features[i] for i in range(17))
        modules2 = list(vgg.features[i] for i in range(17, 24))
        modules3 = list(vgg.features[i] for i in range(24, 31))      # delete the last fc layer.
        self.features1 = nn.Sequential(*modules1)
        self.features2 = nn.Sequential(*modules2)
        self.features3 = nn.Sequential(*modules3)
        self.relu = nn.ReLU()
        if params['feature'] == 'fl':
            self.feature_dim = 512
        elif params['feature'] == 'fa':
            self.feature_dim = 25088
        elif params['feature'] == 'ff':
            self.feature_dim = 1280

        self.pre_rank = pre_rank
        self.new_rank = new_rank
        self.drop_out = nn.Dropout(params['dropout_prob'])
        self.relu = nn.ReLU()
        if params['shareAB']:
            self.matrixAB = nn.Parameter(torch.from_numpy((U[0]+U[1])/2).float())
        else:
            self.matrixA = nn.Parameter(torch.from_numpy(U[0]).float())
            self.matrixB = nn.Parameter(torch.from_numpy(U[1]).float())
        self.matrixC = nn.Parameter(torch.from_numpy(U[2]).float())
        self.fc2 = params['fc2']
        self.params = params

        if self.fc2:
            # two fully connected layers
            self.s_fc1 = nn.Linear(in_features=self.feature_dim, out_features=8192, bias=True)
            self.s_fc2 = nn.Linear(in_features=8192, out_features=np.prod(new_rank), bias=True)
        else:
            self.s_fc = nn.Linear(in_features=self.feature_dim, out_features = np.prod(new_rank), bias=True)

        self.mask = mask
        # two fully connected layers end


    def forward(self, input, target=None):
        feature1 = self.features1(input)
        feature2 = self.features2(feature1)
        feature3 = self.features3(feature2)
        if self.params['feature'] == 'fa':
            feature = feature3.view(feature3.shape[0], -1)
        elif self.params['feature'] == 'fl':
            N,C,H,W = feature3.size()
            feature = feature3.view(N,C, H*W)
            feature = feature.permute(0, 2, 1)
            feature = torch.mean(feature, dim=1)
        elif self.params['feature'] == 'ff':

            N1, C1, H1, W1 = feature1.size()
            feature1 = feature1.view(N1, C1, H1*W1)
            feature1 = feature1.permute(0, 2,1)
            feature1 = torch.mean(feature1, dim=1)

            N2, C2, H2, W2 = feature2.size()
            feature2 = feature2.view(N2, C2, H2*W2)
            feature2 = feature2.permute(0, 2,1)
            feature2 = torch.mean(feature2, dim=1)

            N3, C3, H3, W3 = feature3.size()
            feature3 = feature3.view(N3, C3, H3*W3)
            feature3 = feature3.permute(0, 2,1)
            feature3 = torch.mean(feature3, dim=1)

            feature = torch.cat((feature1, feature2, feature3), dim=1)

        # feature = self.relu(feature)

        if self.fc2:
            # two fully connected layers
            output = self.relu(self.s_fc1(self.drop_out(feature.view(feature.shape[0], -1))))
            output = self.relu(self.s_fc2(self.drop_out(output)))
            output = self.drop_out(output.view((-1,)+self.new_rank))
        else:
            # one fully connected layer
            output = self.relu(self.s_fc(self.drop_out((feature.view(feature.shape[0], -1)))))
            output = self.drop_out(output.view((-1,) + self.new_rank))
            # one fully connected layer end
        if self.shareAB:
            output = torch.einsum('xabc,da -> xdbc', output, self.matrixAB)
            output = torch.einsum('xabc,db -> xadc', output, self.matrixAB)
            output = torch.einsum('xabc,dc -> xabd', output, self.matrixC)
        else:
            output = torch.einsum('xabc,da -> xdbc', output, self.matrixA)
            output = torch.einsum('xabc,db -> xadc', output, self.matrixB)
            output = torch.einsum('xabc,dc -> xabd', output, self.matrixC)
        # return output, weight1, weight2, weight3

        # output = torch.einsum('xabc,xda -> xdbc', core, factor_A)
        # output = torch.einsum('xabc,xdb -> xadc', output, factor_B)
        # output = torch.einsum('xabc,xdc -> xabd', output, factor_C)

        return output

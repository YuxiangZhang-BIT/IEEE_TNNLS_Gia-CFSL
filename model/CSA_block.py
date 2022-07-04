from copy import deepcopy
import torch
from torch import nn
from sklearn import preprocessing
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)

    def forward(self, x, source1, source2):
        message = self.attn(x, source1, source2)
        return message


class AttentionalGNN(nn.Module):
    def __init__(self, num_support, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        self.mlp_dis = MLP([num_support, feature_dim, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        nn.init.constant_(self.mlp_dis[-1].bias, 0.0)
        self.mmd = MMD_loss(kernel_type='linear')
    def forward(self, p_nodes_src, p_nodes_tar, dis_nodes_src, dis_nodes_tar):
        flag = 0
        dev = p_nodes_src[0].device
        for i in range(len(p_nodes_src)):
            p_nodes_src[i] = torch.tensor(preprocessing.scale(p_nodes_src[i].squeeze(0).cpu().detach().numpy())).to(dev).unsqueeze(0).transpose(2,1)
            p_nodes_tar[i] = torch.tensor(preprocessing.scale(p_nodes_tar[i].squeeze(0).cpu().detach().numpy())).to(dev).unsqueeze(0).transpose(2,1)
            dis_nodes_src[i] = self.mlp_dis(torch.tensor(preprocessing.scale(dis_nodes_src[i].squeeze(0).cpu().detach().numpy())).unsqueeze(0).transpose(2,1).to(dev))
            dis_nodes_tar[i] = self.mlp_dis(torch.tensor(preprocessing.scale(dis_nodes_tar[i].squeeze(0).cpu().detach().numpy())).unsqueeze(0).transpose(2,1).to(dev))

        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                p_src0, p_src1 = p_nodes_tar[flag], p_nodes_src[flag]
                d_src0, d_src1 = dis_nodes_tar[flag], dis_nodes_src[flag]
            delta0, delta1 = layer(p_nodes_src[flag], p_src0, p_src0), layer(p_nodes_tar[flag], p_src1, p_src1)
            p_nodes_src_temp, p_nodes_tar_temp = torch.einsum('bij,bij->bij', delta0, p_src0), torch.einsum('bij,bij->bij', delta1, p_src1)
            delta0, delta1 = layer(dis_nodes_src[flag], d_src0, p_nodes_src_temp), layer(dis_nodes_tar[flag], d_src1, p_nodes_tar_temp)
            p_nodes_src[0] = p_nodes_src[0] + self.mlp(torch.cat([p_nodes_src[flag], delta0], dim=1))
            p_nodes_tar[0] = p_nodes_tar[0] + self.mlp(torch.cat([p_nodes_tar[flag], delta1], dim=1))
            
            flag += 1

        return self.mmd(p_nodes_src[0].squeeze(0).transpose(1,0), p_nodes_tar[0].squeeze(0).transpose(1,0))
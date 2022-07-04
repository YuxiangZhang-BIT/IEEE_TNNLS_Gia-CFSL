import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import imp
from model.IDE_block import DPGN
import model.CSA_block
import scipy.io as io


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join( './config', 'paviaU.py'),
                    help='config file with parameters of the experiment. '
                            'It is assumed that the config file is placed under the directory ./config')
args = parser.parse_args()

# Hyper Parameters
config = imp.load_source("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
save_path = config['save_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']

patch_size = train_opt['patch_size']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']
CLASS_NUM = train_opt['class_num']
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
lambda_1 = train_opt['lambda_1']
lambda_2 = train_opt['lambda_2']
GPU = config['gpu']
TEST_CLASS_NUM = train_opt['test_class_num'] # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = train_opt['test_lsample_num_per_class'] # the number of labeled samples per class

utils.same_seeds(0)

# load source data
with open(os.path.join(data_path, source_data), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source data
data_train = source_imdb['data'] 
labels_train = source_imdb['Labels']
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape) 
print(source_imdb['Labels'])

## target data
# load target data
test_data = os.path.join(data_path,target_data)
test_label = os.path.join(data_path,target_data_gt)

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    # HalfWidth
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, 
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class, HalfWidth=patch_size//2) 
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column,nTrain


# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

def conv3x3x3_ft(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        FeatureWiseTransformation2d_fw(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

# --- feature-wise transformation layer ---
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  feature_augment = True
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1)*0.3)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1)*0.5)
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.feature_augment and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, patch_size, emb_size):
        super(D_Res_3d_CNN, self).__init__()
        self.in_channel = in_channel
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.layer_second = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[0],
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[1],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))
                                        
    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, 100,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            _, t, c, w, h = x.size()
            s1 = t * c * w * h
            x = self.conv(x)
            x = x.view(x.shape[0],-1)
            s2 = x.size()[1]
        return s1, s2

    def forward(self, x): 
        x = x.unsqueeze(1) 
        x = self.block1(x) 
        x = self.maxpool1(x) 
        x = self.block2(x)
        x = self.maxpool2(x)
        inter = x
        inter = inter.view(inter.shape[0],-1)
        inter = self.layer_second(inter)
        x = self.conv(x)
        x = x.view(x.shape[0],-1) 
        x = self.layer_last(x)
        out = []
        out.append(inter)
        out.append(x)
        return out


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self, patch_size, emb_size):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16,patch_size, emb_size)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'): 
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x) 
        feature = self.feature_encoder(x) 
        return feature


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().to(GPU)
domain_criterion = nn.BCEWithLogitsLoss().to(GPU)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


# run 10 times
nDataSet = 10
acc = np.zeros([nDataSet, 1])
time_list = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None,None,None,None,None
class_map = np.zeros((nDataSet,), dtype = np.object)
feature_emb_cell = np.zeros((nDataSet,), dtype = np.object)
test_labels_cell = np.zeros((nDataSet,), dtype = np.object)
test_labels_end, feature_emb_end = [], []

seeds = [1336, 1330, 1220, 1233, 1229, 1236, 1226, 1235, 1337, 1224]
for iDataSet in range(nDataSet):
    print('emb_size0:',emb_size)
    print('num_generation:',config['num_generation'] )
    print('patch_size:',patch_size )
    print('seeds:', seeds[iDataSet])
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, patch_size=patch_size)
    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, train_opt['batch_task'], GPU)

    IDE_block_src = DPGN(config['num_generation'],
                      train_opt['dropout'],
                      CLASS_NUM * SHOT_NUM_PER_CLASS,
                      CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,
                      config['point_distance_metric'],
                      config['distribution_distance_metric'],
                      emb_size).to(GPU)

    IDE_block_tar = DPGN(config['num_generation'],
                      train_opt['dropout'],
                      CLASS_NUM * SHOT_NUM_PER_CLASS,
                      CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,
                      config['point_distance_metric'],
                      config['distribution_distance_metric'],
                      emb_size).to(GPU)

    CSA_block_att = CSA_block.AttentionalGNN(num_supports, emb_size, ['cross']*config['num_generation']).to(GPU)
    feature_encoder = Network(patch_size, emb_size)
    IDE_block_src_optim = torch.optim.Adam(IDE_block_src.parameters(), lr=LEARNING_RATE, weight_decay=train_opt['weight_decay'])
    IDE_block_tar_optim = torch.optim.Adam(IDE_block_tar.parameters(), lr=LEARNING_RATE, weight_decay=train_opt['weight_decay'])
    CCSA_block_att_optim = torch.optim.Adam(CSA_block_att.parameters(), lr=LEARNING_RATE, weight_decay=train_opt['weight_decay'])
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)#, weight_decay=train_opt['weight_decay'])


    feature_encoder.apply(weights_init)
    IDE_block_src.apply(weights_init)
    IDE_block_tar.apply(weights_init)
    CSA_block_att.apply(weights_init)
    feature_encoder.to(GPU)

    feature_encoder.train()
    IDE_block_src.train()
    IDE_block_tar.train()
    CSA_block_att.train()

    print("Training...")

    last_accuracy = 0.0
    last_accuracy_gnn = 0.0
    best_episdoe = 0
    train_loss = []
    total_hit_src, total_num_src, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.05

    train_start = time.time()
    for episode in range(EPISODE):
        # get few-shot classification samples
        task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_src = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_src = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

        task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS) 
        support_dataloader_tar = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_tar = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

        # sample datas
        supports_src, support_labels_src = support_dataloader_src.__iter__().next()
        querys_src, query_labels_src = query_dataloader_src.__iter__().next() 

        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().next() 
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().next()

        # calculate features
        support_features_src = feature_encoder(supports_src.to(GPU))
        query_features_src = feature_encoder(querys_src.to(GPU)) 
        support_features_tar = feature_encoder(supports_tar.to(GPU), domain='target') 
        query_features_tar = feature_encoder(querys_tar.to(GPU), domain='target') 

        # initialize nodes and edges for dual graph model
        tensors_src = utils.allocate_tensors()
        for key, tensor in tensors_src.items():
            tensors_src[key] = tensor.to(GPU)
        batch_src = [support_features_src[1].unsqueeze(0).unsqueeze(0),support_labels_src.unsqueeze(0).unsqueeze(0),query_features_src[1].unsqueeze(0).unsqueeze(0),query_labels_src.unsqueeze(0).unsqueeze(0)]
        batch_tar = [support_features_tar[1].unsqueeze(0).unsqueeze(0),support_labels_tar.unsqueeze(0).unsqueeze(0),query_features_tar[1].unsqueeze(0).unsqueeze(0),query_labels_tar.unsqueeze(0).unsqueeze(0)]
        support_data_src, support_label_src, query_data_src, query_label_src, all_data_src, all_label_in_edge_src, node_feature_gd_src, \
        edge_feature_gp_src, edge_feature_gd_src = utils.initialize_nodes_edges(batch_src,
                                                                    num_supports,
                                                                    tensors_src,    
                                                                    train_opt['batch_task'],
                                                                    query_labels_src.shape[0],
                                                                    CLASS_NUM,
                                                                    GPU)
        tensors_tar = utils.allocate_tensors()
        for key, tensor in tensors_tar.items():
            tensors_tar[key] = tensor.to(GPU)
        support_data_tar, support_label_tar, query_data_tar, query_label_tar, all_data_tar, all_label_in_edge_tar, node_feature_gd_tar, \
        edge_feature_gp_tar, edge_feature_gd_tar = utils.initialize_nodes_edges(batch_tar,
                                                                    num_supports,
                                                                    tensors_tar,    
                                                                    train_opt['batch_task'],
                                                                    query_labels_tar.shape[0],
                                                                    CLASS_NUM,
                                                                    GPU)
                                                                    
        # calculate prototype
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_src = [support_features_src[i].reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1) for i in range(len(support_features_src))]
            support_proto_tar = [support_features_tar[i].reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1) for i in range(len(support_features_tar))]
        else:
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        # the IDE-block
        second_last_layer_data_src = torch.cat((support_features_src[0],query_features_src[0])).unsqueeze(0)
        last_layer_data_src = torch.cat((support_features_src[1],query_features_src[1])).unsqueeze(0)

        second_last_layer_data_tar = torch.cat((support_features_tar[0],query_features_tar[0])).unsqueeze(0)
        last_layer_data_tar = torch.cat((support_features_tar[1],query_features_tar[1])).unsqueeze(0)
        
        _, _, _, point_nodes_src, distribution_nodes_src = IDE_block_src(second_last_layer_data_src,
                                                                                        last_layer_data_src,
                                                                                        node_feature_gd_src,
                                                                                        edge_feature_gd_src,
                                                                                        edge_feature_gp_src)

        _, _, _, point_nodes_tar, distribution_nodes_tar = IDE_block_tar(second_last_layer_data_tar,
                                                                                        last_layer_data_tar,
                                                                                        node_feature_gd_tar,
                                                                                        edge_feature_gd_tar,
                                                                                        edge_feature_gp_tar)

        # the CSA-block
        cross_att_loss = CSA_block_att(point_nodes_src, point_nodes_tar, distribution_nodes_src, distribution_nodes_tar)
        
        '''few-shot learning'''
        logits_src = euclidean_metric(query_features_src[1], support_proto_src[1])
        f_loss_src = crossEntropy(logits_src, query_labels_src.long().to(GPU))
        logits_tar = euclidean_metric(query_features_tar[1], support_proto_tar[1])
        f_loss_tar = crossEntropy(logits_tar, query_labels_tar.long().to(GPU))
        f_loss = f_loss_src + f_loss_tar
        '''domain alignment'''
        ot_ep, wd_ep, gwd_ep = utils.OT(point_nodes_src, point_nodes_tar)
        ot_ed, wd_ed, gwd_ed = utils.OT(distribution_nodes_src, distribution_nodes_tar)

        domain_loss = ot_ep + ot_ed
                                        
        loss = f_loss + lambda_1*domain_loss+ lambda_2*cross_att_loss  # 0.01

        # Update parameters
        feature_encoder.zero_grad()
        IDE_block_src_optim.zero_grad()
        IDE_block_tar_optim.zero_grad()
        CCSA_block_att_optim.zero_grad()
        loss.backward()
        feature_encoder_optim.step()
        IDE_block_src_optim.step()
        IDE_block_tar_optim.step()
        CCSA_block_att_optim.step()

        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_labels_src).item()
        total_num_src += querys_src.shape[0]
        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_labels_tar).item()
        total_num_tar += querys_tar.shape[0]

        if (episode + 1) % 100 == 0:  
            train_loss.append(loss.item())
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, cross att loss: {:6.4f}, acc_src {:6.4f}, acc_tar {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                domain_loss.item(),
                                                                                                                f_loss.item(),
                                                                                                                cross_att_loss.item(),
                                                                                                                total_hit_src / total_num_src,
                                                                                                                total_hit_tar / total_num_tar,
                                                                                                                loss.item()))
            print('fea_lr： {:6.6f}'.format(feature_encoder_optim.param_groups[0]['lr']))
            print('wd {:6.6f}, gwd: {:6.6f}'.format((wd_ed+wd_ep).item(), (gwd_ed+gwd_ep).item()))
        if (episode + 1) % 500 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            predict_gnn = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = train_loader.__iter__().next()
            _, train_features = feature_encoder(Variable(train_datas).to(GPU), domain='target')

            max_value = train_features.max()
            min_value = train_features.min() 
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            test_labels_all, feature_emb= [], []
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                _, test_features = feature_encoder(Variable(test_datas).to(GPU), domain='target')
                feature_emb.append(test_features.cpu().detach().numpy())
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())

                test_labels = test_labels.numpy()
                test_labels_all.append(test_labels)
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            print('seeds:', seeds[iDataSet])
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
                best_predict_all = predict
                best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                feature_emb_end = np.concatenate(feature_emb)
                test_labels_end = np.concatenate(test_labels_all)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('iter:{} best episode:[{}], best accuracy_gnn={}'.format(iDataSet, best_episdoe + 1, last_accuracy_gnn))
    print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
    print("accuracy list: ", acc)
    print('***********************************************************************************')
    for i in range(len(best_predict_all)):
        best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1
    class_map[iDataSet] = best_G[4:-4, 4:-4]
    time_list[iDataSet] = train_end-train_start
    feature_emb_cell[iDataSet] = feature_emb_end
    test_labels_cell[iDataSet] = test_labels_end

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print("accuracy list: ", acc)
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))
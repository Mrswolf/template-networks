# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainda.algorithms.deep_learning.base import (
    compute_out_size, compute_same_pad2d, 
    MaxNormConstraintConv2d, MaxNormConstraintLinear,
    SkorchNet)

@SkorchNet
class FTN(nn.Module):
    def __init__(self,
        n_bands, n_features,
        n_channels, n_samples, n_classes,
        templates,
        band_kernel=9, pooling_kernel=2,
        eps=1e-8, dropout=0.95):
        super().__init__()
        self.n_bands = n_bands
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.eps = torch.tensor(eps)
        templates = torch.as_tensor(templates, dtype=torch.float)
        templates = templates.unsqueeze(1)
        self.register_buffer('templates', templates)
        self.register_buffer('template_ind', torch.arange(n_classes, dtype=torch.long))
        
        self.feature_extractor_X = nn.Sequential(OrderedDict([
            ('same_pad1', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (n_channels, n_samples), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('spatial_layer', nn.Conv2d(
                n_bands, n_features, (n_channels, 1), 
                stride=(1, 1), bias=False)),
            ('temporal_layer1', nn.Conv2d(
                n_features, n_features, (1, pooling_kernel),
                stride=(1, pooling_kernel), padding=(0, 0), bias=False)),
            ('bn_layer', nn.BatchNorm2d(n_features)),
            ('tanh_layer', nn.Tanh()),
            ('same_pad2', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (1, compute_out_size(n_samples, pooling_kernel, stride=pooling_kernel)), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('temporal_layer2', nn.Conv2d(
                n_features, n_features, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
        ]))
        
        self.feature_extractor_Yf = nn.Sequential(OrderedDict([
            ('same_pad1', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (templates.shape[2], n_samples), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('spatial_layer', nn.Conv2d(
                n_bands, n_features, (templates.shape[2], 1), 
                stride=(1, 1), bias=False)),
            ('temporal_layer1', nn.Conv2d(
                n_features, n_features, (1, pooling_kernel),
                stride=(1, pooling_kernel), padding=(0, 0), bias=False)),
            ('bn_layer', nn.BatchNorm2d(n_features)),
            ('tanh_layer', nn.Tanh()),
            ('same_pad2', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (1, compute_out_size(n_samples, pooling_kernel, stride=pooling_kernel)), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('temporal_layer2', nn.Conv2d(
                n_features, n_features, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
        ]))        
        
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=eps)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            X = torch.zeros(1, 1, n_channels, n_samples)
            T = self.templates
            X = self.feature_extractor_X(X)
            T = self.feature_extractor_Yf(T)
            X = torch.reshape(X, (X.shape[0], 1, X.shape[1], -1))
            T = torch.reshape(T, (1, T.shape[0], T.shape[1], -1))
            out = self.cosine_similarity(X, T)
            out = self.flatten(out)
        
        self.fc_layer = nn.Linear(out.shape[-1], n_classes)
        self.fc_drop = nn.Dropout(dropout)
        self.instance_norm = nn.InstanceNorm2d(1)
      
    def forward(self, X, y=None):
        X = X.unsqueeze(1)
        X = self.instance_norm(X)
        T = self.templates
        X = self.feature_extractor_X(X)
        T = self.feature_extractor_Yf(T)
        X = torch.reshape(X, (X.shape[0], 1, X.shape[1], -1))
        T = torch.reshape(T, (1, T.shape[0], T.shape[1], -1))
        out = self.cosine_similarity(X, T)
        out = self.flatten(out)
        out = self.fc_drop(out)
        out = self.fc_layer(out)
        return out
    

@SkorchNet
class DTN(nn.Module):
    def __init__(self,
        n_bands, n_features,
        n_channels, n_samples, n_classes,
        band_kernel=9,
        pooling_kernel=2, 
        dropout=0.5, momentum=None, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('encode_table', torch.sparse.torch.eye(n_classes, dtype=torch.long))
        
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('same1', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (n_channels, n_samples), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('spatial_layer', MaxNormConstraintConv2d(
                n_bands, n_features, (n_channels, 1), 
                stride=(1, 1), bias=False, max_norm_value=1)),
            ('temporal_layer1', nn.Conv2d(
                n_features, n_features, (1, pooling_kernel),
                stride=(1, pooling_kernel), padding=(0, 0), bias=False)),
            ('bn2', nn.BatchNorm2d(n_features)),
            ('tanh2', nn.Tanh()),
            ('same2', nn.ConstantPad2d(
                    compute_same_pad2d(
                        (1, compute_out_size(n_samples, pooling_kernel, stride=pooling_kernel)), 
                        (1, band_kernel), 
                        stride=(1, 1)), 
                    0)),
            ('temporal_layer2', nn.Conv2d(
                n_features, n_features, (1, band_kernel),
                stride=(1, 1), padding=(0, 0), bias=False)),
        ]))
        
        self.cosine_similarity = nn.CosineSimilarity(
            dim=-1, eps=self.eps)
        self.flatten = nn.Flatten()
        self.fc_drop = nn.Dropout(dropout)
        
        with torch.no_grad():
            X = torch.zeros(1, 1, n_channels, n_samples)
            X = self.feature_extractor(X)
            self._register_templates(n_classes, *X.shape[1:])
            X = torch.reshape(
                X, (X.shape[0], 1, X.shape[1], -1))
            T = self.running_template
            T = torch.reshape(
                T, (1, T.shape[0], T.shape[1], -1))
            out = self.cosine_similarity(X, T)
            out = self.flatten(out)
            
        self.fc_layer = nn.Linear(out.shape[-1], n_classes)
        self.instance_norm = nn.InstanceNorm2d(1)
        
    def _register_templates(self, n_classes, *args):
        self.register_buffer(
            'running_template', torch.zeros(n_classes, *args))
        nn.init.xavier_uniform_(self.running_template, gain=1)
    
    def _update_templates(self, X, y):
        # update templates
        with torch.no_grad():
            sampleX = X
            sampleY = y
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum
            
            mask = torch.index_select(self.encode_table, 0, sampleY)
            N = torch.maximum(torch.sum(mask, 0), torch.tensor(self.eps))
            # supervised feature-template
            features = self.feature_extractor(sampleX)
            mask_data = torch.reshape(mask, (*mask.shape, *[1 for _ in range(len(features.shape)-1)])) * torch.unsqueeze(features, 1)
            # n_classes, ...
            new_template = torch.sum(mask_data, 0) / torch.reshape(N, (-1, *[1 for _ in range(len(features.shape)-1)])) 
            self.running_template = (
                (1-exponential_average_factor) * self.running_template
                + exponential_average_factor * new_template)
            
    def forward(self, X, y=None):
        X = X.unsqueeze(1)
        X = self.instance_norm(X)
        out = self.feature_extractor(X)
        out = torch.reshape(out, (out.shape[0], 1, out.shape[1], -1))
        T = self.running_template
        T = torch.reshape(T, (1, T.shape[0], T.shape[1], -1))
        out = self.cosine_similarity(out, T)
        out = self.flatten(out)
        out = self.fc_drop(out)
        out = self.fc_layer(out)
        if self.training:
            self._update_templates(X, y)
        return out

    def update_running_templates(self, templates):
        with torch.no_grad():
            self.running_template.zero_()
            self.running_template.add_(templates)
            
    def reset_statistics(self):
        with torch.no_grad():
            self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
            
            
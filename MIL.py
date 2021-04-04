# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class feature_extractor(nn.Module):
    def __init__(self, encoder):
        super(feature_extractor, self).__init__()
        self.feature_ex = encoder

    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.view(feature.size(0), -1)
        return feature

class class_predictor(nn.Module):
    def __init__(self):
        super(class_predictor, self).__init__()
        # 次元圧縮
        # ResNet50(特徴ベクトル長: 2048 -> 512)
        self.feature_extractor_2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # class classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 3),
        )

    def forward(self, input):
        x = input.squeeze(0)
        H = self.feature_extractor_2(x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)  # KxL
        class_prob = self.classifier(M)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))
        return class_prob, class_hat

class MIL(nn.Module):
    def __init__(self, feature_ex, class_predictor):
        super(MIL, self).__init__()
        self.feature_extractor = feature_ex
        self.class_predictor = class_predictor
    
    def forward(self, input):
        x = input.squeeze(0)
        # 特徴ベクトル抽出
        features = self.feature_extractor(x)
        # class分類
        class_prob, class_hat = self.class_predictor(features)
        
        return class_prob, class_hat

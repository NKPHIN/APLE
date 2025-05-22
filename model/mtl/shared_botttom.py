"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

    Weichen Shen, weichenswc@163.com

Reference:
    [1] Ruder S. An overview of multi-task learning in deep neural networks[J]. arXiv preprint arXiv:1706.05098, 2017.(https://arxiv.org/pdf/1706.05098.pdf)
"""


import torch
import torch.nn as nn
from torch.nn import functional as F

class SharedBottomModel(nn.Module):

    # 相当于一个共享专家
    def __init__(self, user_feature_dict,item_feature_dict,emb_dim = 128,tower_hidden_dim = [128,128],
                 bottom_hidden_dim = [128,128],activation  = nn.ReLU(),nums_task  = 2,device = "cuda"):
        
        
        super(SharedBottomModel, self).__init__()
        if user_feature_dict is None and item_feature_dict is None:
            raise Exception("user_feature_dict and item_feature_dict cannot be None at the same time")
        if not isinstance(user_feature_dict,dict) or not isinstance(item_feature_dict,dict):
            raise Exception("user_feature_dict and item_feature_dict should be dict")
        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.emb_dim = emb_dim
        self.tower_hidden_dim = tower_hidden_dim
        self.bottom_hidden_dim = bottom_hidden_dim
        self.activation = activation
        self.nums_task = nums_task
        self.device = device
        
        #初始化 embeding
        user_cate_feature_nums,item_cate_feature_nums = 0,0
        for user_cate ,num in self.user_feature_dict.items():
            if num[0]>1:
                user_cate_feature_nums += 1
                setattr(self,user_cate,nn.Embedding(num[0],self.emb_dim))
        for item_cate,num in self.item_feature_dict.items():
            if num[0]>1:
                item_cate_feature_nums += 1
                setattr(self,item_cate,nn.Embedding(num[0],self.emb_dim))

        hidden_size = self.emb_dim*(user_cate_feature_nums+item_cate_feature_nums)+\
                        (len(self.user_feature_dict)-user_cate_feature_nums+len(self.item_feature_dict)-item_cate_feature_nums)
                      
        #bottom_tower，使用 bottomhiddensize
        self.bottom_tower = nn.Sequential(
            nn.Linear(hidden_size,self.bottom_hidden_dim[0]),
            self.activation,
            nn.Linear(self.bottom_hidden_dim[0],self.bottom_hidden_dim[1]),
            self.activation
        )

        #任务塔
        for i in range(self.nums_task):
            setattr(self,f"task_tower{i+1}",nn.ModuleList())
            for j in range(len(tower_hidden_dim)):
                getattr(self,f"task_tower{i+1}").add_module(f"hidden_layer{j+1}",
                                                                    nn.Linear(self.bottom_hidden_dim[-1],tower_hidden_dim[j]))
                getattr(self,f"task_tower{i+1}").add_module(f"activate{j+1}",self.activation)
                getattr(self,f"task_tower{i+1}").add_module(f"batchnorm{j+1}",nn.BatchNorm1d(tower_hidden_dim[j]))
                getattr(self,f"task_tower{i+1}").add_module(f"dropout{j+1}",nn.Dropout(0.5))
            getattr(self,f"task_tower{i+1}").add_module(f"output_layer",nn.Linear(tower_hidden_dim[-1],1))
 
    def forward(self,x):
        assert x.size()[1]==len(self.user_feature_dict)+len(self.item_feature_dict)
        
        # 从嵌入里找对应的特征
        user_embed_list,item_embed_list = [],[]
        for user_cate,num in self.user_feature_dict.items():
            if num[0]>1:
                user_embed_list.append(getattr(self,user_cate)(x[:,num[1]].long()))
            else:
                user_embed_list.append(x[:,num[1]].unsqueeze(1))
        for item_cate,num in self.item_feature_dict.items():
            if num[0]>1:
                item_embed_list.append(getattr(self,item_cate)(x[:,num[1]].long()))
            else:
                item_embed_list.append(x[:,num[1]].unsqueeze(1))

        #拼接特征
        user_embed = torch.cat(user_embed_list,dim=1)
        item_embed = torch.cat(item_embed_list,dim=1)

        hidden  = torch.cat([user_embed,item_embed],dim=1).float()

        #bottom_tower
        hidden = self.bottom_tower(hidden)

        #任务塔
        output_list = []
        for i in range(self.nums_task):
            output = hidden
            for mod in getattr(self,f"task_tower{i+1}"):
                output = mod(output)
            output_list.append(output)
        
        return output_list
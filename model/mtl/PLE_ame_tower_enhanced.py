import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings("ignore")


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# tower的实现
class Block(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Block, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class HiddenLayer(nn.Module):
    def __init__(self, tower_size, input_size, hidden_size, output_size):
        super(HiddenLayer, self).__init__()
        self.tower_size = tower_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.blocks = nn.ModuleList([Block(self.input_size, self.output_size, self.hidden_size) for i in range(self.tower_size)])

    def forward(self, x):
        output = [block(_x) for _x, block in zip(x, self.blocks)]

        for i in range(len(output)):
            if i == 0:
                continue
            output[i] = output[i] + output[i-1]

        return output 

class Tower(nn.Module):
    def __init__(self, tower_size, input_size, hidden_size):
        super(Tower, self).__init__()
        self.tower_size = tower_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layer_1 = HiddenLayer(self.tower_size, self.input_size, self.hidden_size, self.hidden_size)
        self.layer_2 = HiddenLayer(self.tower_size, self.hidden_size, self.hidden_size, self.hidden_size)
        self.layer_3 = HiddenLayer(self.tower_size, self.hidden_size, self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output_1 = self.layer_1(x)
        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_2)
        
        # 因为使用BCEwithlogits，所以不需要sigmoid
        # out = [self.sigmoid(res) for res in output_3]
        out = output_3

        return out



class PLE_Hiddle_Layer(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden):
        super(PLE_Hiddle_Layer, self).__init__()
        self.input_size = input_size # input_size 是一个数组，包含num_task+1个input_size 
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden

        is_first = isinstance(self.input_size, list)
        if is_first:
            self.experts_shared = nn.ModuleList([Expert(self.input_size[-1], self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
            self.experts_task1 = nn.ModuleList([Expert(self.input_size[0], self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
            self.experts_task2 = nn.ModuleList([Expert(self.input_size[1], self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
            self.soft = nn.Softmax(dim=1)
            self.dnn1 = nn.Sequential(nn.Linear(self.input_size[0], self.num_specific_experts+self.num_shared_experts),
                                    nn.Softmax(dim=1))
            self.dnn2 = nn.Sequential(nn.Linear(self.input_size[1], self.num_specific_experts + self.num_shared_experts),
                                    nn.Softmax(dim=1))
            self.dnn_share = nn.Sequential(nn.Linear(self.input_size[2], self.num_specific_experts * 2 + self.num_shared_experts),
                                        nn.Softmax(dim=1))
        
        else:
            self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
            self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
            self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
            self.soft = nn.Softmax(dim=1)
            self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts+self.num_shared_experts),
                                    nn.Softmax(dim=1))
            self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                    nn.Softmax(dim=1))
            self.dnn_share = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts * 2 + self.num_shared_experts),
                                        nn.Softmax(dim=1))


    # 对于第一层，输入相同，即都是 input 
    def forward(self, x1, x2, x_share):
        experts_shared_o = [e(x_share) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)
        experts_task1_o = [e(x1) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)
        experts_task2_o = [e(x2) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        # gate1
        selected1 = self.dnn1(x1)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)

        # gate2
        selected2 = self.dnn2(x2)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)

        # gate_share
        selected_share = self.dnn_share(x_share)
        gate_expert_ouput_share = torch.cat((experts_task1_o, experts_task2_o, experts_shared_o), dim=0)
        gate_share_out = torch.einsum('abc, ba -> bc', gate_expert_ouput_share, selected_share)

        return [gate1_out, gate2_out, gate_share_out]


class PLE_Output_Layer(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden):
        super(PLE_Output_Layer, self).__init__()
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.soft = nn.Softmax(dim=1)
        self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts+self.num_shared_experts),
                                 nn.Softmax())
        self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax())
        self.tower = Tower(2, self.experts_out, self.towers_hidden)


    def forward(self, x1, x2, x_share):
        experts_shared_o = [e(x_share) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)
        experts_task1_o = [e(x1) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)
        experts_task2_o = [e(x2) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        # gate1
        selected1 = self.dnn1(x1)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        # final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.dnn2(x2)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        # final_output2 = self.tower2(gate2_out)

        final_out = self.tower([gate1_out, gate2_out])
        return final_out



class PLE(nn.Module):
    
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden):
        super(PLE, self).__init__()
        self.input_size = input_size # input_size 是一个数组，包含num_task+1个input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        
        self.hidden1 = PLE_Hiddle_Layer(input_size=input_size, num_specific_experts=num_specific_experts, num_shared_experts=num_shared_experts,
                                       experts_out=experts_out, experts_hidden=experts_hidden)
        self.hidden2 = PLE_Hiddle_Layer(input_size=experts_out, num_specific_experts=num_specific_experts, num_shared_experts=num_shared_experts,
                                       experts_out=experts_out, experts_hidden=experts_hidden)
        self.output = PLE_Output_Layer(input_size=experts_out, num_specific_experts=num_specific_experts, num_shared_experts=num_shared_experts,
                                       experts_out=experts_out, experts_hidden=experts_hidden, towers_hidden=towers_hidden)


    def forward(self, x):
        x1, x2, x_share = self.hidden1(x[0], x[1], x[2])
        x1, x2, x_share = self.hidden2(x1, x2, x_share)
        final_output1, final_output2 = self.output(x1, x2, x_share)

        return [final_output1, final_output2]
    

class PLE_final_AME_Tower_enhanced(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim, hidden_dim = 128,
                 output_size=1, num_task=2,device = 'cuda',multi_embedding = [32,128,64]):
        super(PLE_final_AME_Tower_enhanced, self).__init__()
        
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        # embedding初始化,num_task+1个嵌入矩阵
        
        for i in range(num_task):
            user_cate_feature_nums, item_cate_feature_nums = 0, 0
            for user_cate, num in self.user_feature_dict.items():
                if num[0] > 1:
                    user_cate_feature_nums += 1
                    setattr(self, f"{user_cate}_task{i+1}", nn.Embedding(num[0], multi_embedding[i]))
            for item_cate, num in self.item_feature_dict.items():
                if num[0] > 1:
                    item_cate_feature_nums += 1
                    setattr(self, f"{item_cate}_task{i+1}", nn.Embedding(num[0], multi_embedding[i]))
        
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, f"{user_cate}_shared", nn.Embedding(num[0], multi_embedding[-1]))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, f"{item_cate}_shared", nn.Embedding(num[0], multi_embedding[-1]))

        
        # user embedding + item embedding
        # 输出最终的 num_task+1 个hidden_size
        hidden_size = []
        for i in range(num_task):
            hidden_size.append(multi_embedding[i]*(user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(user_feature_dict) - user_cate_feature_nums) + (
                                  len(item_feature_dict) - item_cate_feature_nums))
        hidden_size.append(multi_embedding[-1]*(user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(user_feature_dict) - user_cate_feature_nums) + (
                                  len(item_feature_dict) - item_cate_feature_nums))
        
        

        #ple 第一层需要处理不同的输入,hidden_sizes是一个数组，包含num_task+1个hidden_size
        self.ple = PLE(input_size=hidden_size, num_specific_experts=2, num_shared_experts=2, experts_out=hidden_dim, experts_hidden=hidden_dim, towers_hidden=hidden_dim)

    
    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # 
        final_hidden = []
        for i in range(self.num_task+1):
            if i== self.num_task:
                # 共享expert的嵌入
                user_embed_list, item_embed_list = list(), list()
                for user_feature, num in self.user_feature_dict.items():
                    if num[0] > 1:
                        user_embed_list.append(getattr(self, f"{user_feature}_shared")(x[:, num[1]].long()))    # num[1] 表示该列在张量中的列索引
                    else:
                        user_embed_list.append(x[:, num[1]].unsqueeze(1))
                for item_feature, num in self.item_feature_dict.items():
                    if num[0] > 1:
                        item_embed_list.append(getattr(self, f"{item_feature}_shared")(x[:, num[1]].long()))
                    else:
                        item_embed_list.append(x[:, num[1]].unsqueeze(1))

                # embedding 融合
                user_embed = torch.cat(user_embed_list, axis=1)
                item_embed = torch.cat(item_embed_list, axis=1)
                hidden = torch.cat([user_embed, item_embed], axis=1).float()
                final_hidden.append(hidden)
            else:
                #specific—task的嵌入
                user_embed_list, item_embed_list = list(), list()
                for user_feature, num in self.user_feature_dict.items():
                    if num[0] > 1:
                        user_embed_list.append(getattr(self, f"{user_feature}_task{i+1}")(x[:, num[1]].long()))
                    else:
                        user_embed_list.append(x[:, num[1]].unsqueeze(1))
                for item_feature, num in self.item_feature_dict.items():
                    if num[0] > 1:
                        item_embed_list.append(getattr(self, f"{item_feature}_task{i+1}")(x[:, num[1]].long()))
                    else:
                        item_embed_list.append(x[:, num[1]].unsqueeze(1))

                # embedding 融合
                user_embed = torch.cat(user_embed_list, axis=1)
                item_embed = torch.cat(item_embed_list, axis=1)

                # hidden layer
                hidden = []
                
                hidden = torch.cat([user_embed, item_embed], axis=1).float()
                final_hidden.append(hidden)
            
        
        output = self.ple(final_hidden)

        return output


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

random.seed(3)
np.random.seed(3)
seed = 3
batch_size = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    

class PLE_Hiddle_Layer(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden):
        super(PLE_Hiddle_Layer, self).__init__()
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])
        self.soft = nn.Softmax(dim=1)
        self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts+self.num_shared_experts),
                                 nn.Softmax())
        self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax())
        self.dnn_share = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts * 2 + self.num_shared_experts),
                                       nn.Softmax());


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
        self.tower1 = Tower(self.experts_out, 1, self.towers_hidden)
        self.tower2 = Tower(self.experts_out, 1, self.towers_hidden)


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
        final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.dnn2(x2)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)

        return [final_output1, final_output2]



class PLE(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, towers_hidden):
        super(PLE, self).__init__()
        self.input_size = input_size
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
        x1, x2, x_share = self.hidden1(x, x, x)
        x1, x2, x_share = self.hidden2(x1, x2, x_share)
        final_output1, final_output2 = self.output(x1, x2, x_share)

        return [final_output1, final_output2]


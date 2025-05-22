import torch
import random
import time
import argparse
from utils import *
from trainer import *
from neg_sampler import *
from load_model import *
from splitter import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


from model.mtl.esmm import ESMM
from model.mtl.mmoe import MMOE
from model.mtl.shared_botttom import SharedBottomModel
from model.mtl.PLE_original import PLE_final
from model.mtl.PLE_ame_tower_enhanced import PLE_final_AME_Tower_enhanced

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def select_sampler(train_data, val_data, test_data, user_count, item_count, args):
    if args.sample == 'random':
        return RandomNegativeSampler(train_data, val_data, test_data, user_count, item_count, args.negsample_size, args.seed, args.negsample_savefolder)
    elif args.sample == 'popular':
        return PopularNegativeSampler(train_data, val_data, test_data, user_count, item_count, args.negsample_size, args.seed, args.negsample_savefolder)

def get_data(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    
    if name == 'mtl':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = mtl_data(path, args)
        if args.mtl_task_num == 2:
            train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
        else:
            train_dataset = (train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values)
        train_dataset = mtlDataSet(train_dataset, args)
        val_dataset = mtlDataSet(val_dataset, args)
        test_dataset = mtlDataSet(test_dataset, args)

        # dataloader
        train_dataloader = get_train_loader(train_dataset, args)
        val_dataloader = get_val_loader(val_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)

        return train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict
    else:
        raise ValueError('unknown dataset name: ' + name)

# def get_model(args, linear_feature_columns=None, dnn_feature_columns=None, history_feature_list=None):
#     name = args.model_name
#     if name == 'deepfm':
#         return DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'nfm':
#         return NFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'xdeepfm':
#         return xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'wdl':
#         return WDL(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'afm':
#         return AFM(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'dcn':
#         return DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'dcnmix':
#         return DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', device=args.device)
#     elif name == 'din':
#         return DIN(dnn_feature_columns, history_feature_list, task='binary', device=args.device)
#     elif name == 'dien':
#         return DIEN(dnn_feature_columns, history_feature_list, task='binary', device=args.device)
#     elif name == 'bert4rec':
#         return BERTModel(args)
#     elif name == 'sasrec':
#         return SASRec(args)
#     elif name == 'nextitnet':
#         return NextItNet(args)
#     elif name == 'gru4rec':
#         return GRU4Rec(args)
#     elif name == 'peterrec':
#         return PeterRec(args)
#     elif name == 'stackrec':
#         return StackRec(args)
#     elif name == 'cprec':
#         return CpRec(args)
#     elif name == 'skiprec':
#         return SkipRec(args), PolicyNetGumbel(args)
#     elif name == 'sas4infacc':
#         return SAS4infaccModel(args), SAS_PolicyNetGumbel(args)
#     elif name == 'sas4transfer':
#         return SAS_TransferModel(args)
#     elif name == 'bert4profile':
#         return BERT_ProfileModel(args)
#     elif name == 'peter4profile':
#         return Peter_ProfileModel(args)
#     elif name == 'conure':
#         return Conure(args)
#     elif name == 'bert4life':
#         return BERT4Life(args)
#     elif name == 'sas4life':
#         return SAS4Life
#     elif name == 'bert4coldstart':
#         return BERT_ColdstartModel(args)
#     elif name == 'peter4coldstart':
#         return Peter4Coldstart(args)
#     elif name == 'dnn4profile':
#         return DNNModel(args)
#     elif name == 'sas4acc':
#         return SAS4accModel(args)
#     elif name == 'sas4cp':
#         return SAS4cpModel(args)
#     elif name == 'ncf':
#         return NCF(args)
#     elif name == 'mf':
#         return MF(args)
#     elif name == 'lightgcn':
#         return LightGCN(args)
#     elif name == 'ngcf':
#         return NGCF(args)
#     # elif name == 'vae':
#     #     return VAECF(args)
#     # elif name == 'item2vec':
#     #     return Item2Vec(args)
#     else:
#         raise ValueError('unknown model name: ' + name)

def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', default='')
    parser.add_argument('--task_num', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--sample', type=str, default='random')
    parser.add_argument('--negsample_savefolder', type=str, default='./data/neg_data/')
    parser.add_argument('--negsample_size', type=int, default=99)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--item_min', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    # parser.add_argument('--save_path', type=str, default='/data/home')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--valid_rate', type=int, default=100)

    parser.add_argument('--model_name', default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--re_epochs', type=int, default=20)

    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--device', default='cuda')  # cuda:0
    parser.add_argument('--is_parallel', type=bool, default=False)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='l2 regularization') #0.008
    parser.add_argument('--decay_step', type=int, default=5, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR')
    parser.add_argument('--num_users', type=int, default=1, help='Number of total users')
    parser.add_argument('--num_items', type=int, default=1, help='Number of total items')
    parser.add_argument('--num_embedding', type=int, default=1, help='Number of total source items')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of total labels')
    parser.add_argument('--k', type=int, default=20, help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)')
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 20], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

    #model param
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden vectors (model)')
    parser.add_argument('--block_num', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of transformer groups')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for multi-attention')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.3,
                        help='Probability for masking items in the training sequence')
    parser.add_argument('--factor_num', type=int, default=128)
    #Nextitnet
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size for model')
    parser.add_argument('--dilations', type=int, default=[1, 4], help='Number of transformer layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Number of heads for multi-attention')
    parser.add_argument('--is_mp', type=bool, default=False, help='Number of heads for multi-attention')
    parser.add_argument('--pad_token', type=int, default=0)
    parser.add_argument('--temp', type=int, default=7)

    #SASRec
    parser.add_argument('--l2_emb', default=0.0, type=float)
    #mtl
    parser.add_argument('--mtl_task_num', type=int, default=1, help='0:like, 1:click, 2:two tasks')

    #CF
    parser.add_argument('--test_method', default='ufo', type=str)
    parser.add_argument('--val_method', default='ufo', type=str)
    parser.add_argument('--test_size', default=0.1, type=float)
    parser.add_argument('--val_size', default=0.1111, type=float)
    parser.add_argument('--cand_num', default=100, type=int)
    parser.add_argument('--sample_method', default='high-pop', type=str) #
    # parser.add_argument('--sample_method', default='uniform', type=str)
    parser.add_argument('--sample_ratio', default=0.3, type=float)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--loss_type', default='BPR', type=str)
    parser.add_argument('--init_method', default='default', type=str)
    parser.add_argument('--optimizer', default='default', type=str)
    parser.add_argument('--early_stop', default=True, type=bool)
    parser.add_argument('--reg_1', default=0.0, type=float)
    parser.add_argument('--reg_2', default=0.0, type=float)
    parser.add_argument('--context_window', default=2, type=int)
    parser.add_argument('--rho', default=0.5, type=float)

    #ngcf
    parser.add_argument('--node_dropout', default=0.1,
                        type=float,
                        help='NGCF: Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', default=0.1,
                        type=float,
                        help='NGCF: Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--hidden_size_list', default=[128, 128], type=list)

    #vae
    parser.add_argument('--latent_dim',
                        type=int,
                        default=128,
                        help='bottleneck layer size for autoencoder')
    parser.add_argument('--anneal_cap',
                        type=float,
                        default=0.2,
                        help='Anneal penalty for VAE KL loss')
    parser.add_argument('--total_anneal_steps',
                        type=int,
                        default=1000)
    #model_KD
    parser.add_argument('--kd', type=bool, default=False, help='True: Knowledge distilling, False: Cprec')
    parser.add_argument('--alpha', default=0.4, type=float)

    #model_acc
    parser.add_argument('--add_num_times', type=int, default=2)

    #transfer learning
    parser.add_argument('--is_pretrain', type=int, default=1, help='0: mean transfer, 1: mean pretrain, 2:mean train full model without transfer')

    #user_profile_represent
    parser.add_argument('--user_profile', type=str, default='gender', help='user_profile: gender, age')

    # life_long
    parser.add_argument('--prun_rate', type=float, default=0)
    parser.add_argument('--ll_max_itemnum', type=int, default=0)
    parser.add_argument('--lifelong_eval', type=bool, default=True)
    parser.add_argument('--task1_out', type=int, default=0)
    parser.add_argument('--task2_out', type=int, default=0)
    parser.add_argument('--task3_out', type=int, default=0)
    parser.add_argument('--task4_out', type=int, default=0)
    parser.add_argument('--eval', type=bool, default=True)

    # cold_start
    parser.add_argument('--ch', type=bool, default=True)

    args = parser.parse_args()
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)
    # if 'bert' in args.model_name:
    set_seed(args.seed)
    writer = SummaryWriter()
    print(args)
    
    if args.task_name == 'mtl':

        train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)
        if args.mtl_task_num == 2:
            num_task = 2
        else:
            num_task = 1
        if args.model_name == 'esmm':
            model = ESMM(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, num_task=num_task)
        elif args.model_name == 'mmoe':
            model = MMOE(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
        elif args.model_name == 'shared_bottom':
            model = SharedBottomModel(user_feature_dict, item_feature_dict,emb_dim=args.embedding_size, device=args.device, nums_task=num_task)
        else:
            # model = PLE_final(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
            model = PLE_final_AME_Tower_enhanced(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
 
        mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args, train=False)
    





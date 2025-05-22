import torch
import random
import time
import argparse
from utils import *
from trainer import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


from model.mtl.esmm import ESMM
from model.mtl.mmoe import MMOE
from model.mtl.shared_botttom import SharedBottomModel
from model.mtl.PLE_original import PLE_final
from model.mtl.PLE_ame_tower_enhanced import PLE_final_AME_Tower_enhanced

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# default to ensure reproducibility
def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_data_loader(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    
    if name == 'mtl':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = read_mtl_data(path, args)
        
        if args.mtl_task_num == 2:
            # extract data from DataFrame to Tuple (features, label_click, lable_like)
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
        raise ValueError('unknown task name: ' + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add params
    parser.add_argument('--device', default='cuda') 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', default='')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--save_path', type=str, default='/data/home')
    parser.add_argument('--model_name', default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size for model')
    parser.add_argument('--mtl_task_num', type=int, default=1, help='0:like, 1:click, 2:two tasks')

    args = parser.parse_args()
    device = torch.device(args.device)   # default local cuda 0
    set_seed(args.seed)
    writer = SummaryWriter()
    print(args)
    
    if args.task_name == 'mtl':

        train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data_loader(args)
        num_task = args.mtl_task_num     # default 1 task

        if args.model_name == 'esmm':
            model = ESMM(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, num_task=num_task)
        elif args.model_name == 'mmoe':
            model = MMOE(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
        elif args.model_name == 'shared_bottom':
            model = SharedBottomModel(user_feature_dict, item_feature_dict,emb_dim=args.embedding_size, device=args.device, nums_task=num_task)
        else:
            model = PLE_final_AME_Tower_enhanced(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
 
        mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args, train=True)    # train & validation
        mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args, train=False)   # test
    





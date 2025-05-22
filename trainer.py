import os.path
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import os


def train_single_task(model, train_loader, val_loader, model_path, args):
    device = args.device
    epoch = args.epochs
    early_stop = 5
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)

    patience, eval_loss = 0, 0
    for i in range(epoch):
        model.train()
        y_train_label_true = []
        y_train_label_predict = []
        total_loss, count = 0, 0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            predict = model(x)
            y_train_label_true += list(y.squeeze().cpu().numpy())
            y_train_label_predict += list(predict[0].squeeze().cpu().detach().numpy())

            loss = loss_function(predict[0], y.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        
        auc = roc_auc_score(y_train_label_true, y_train_label_predict)
        print("Epoch %d train loss is %.3f, auc is %.3f" % (i + 1, total_loss / count, auc))

        # validation
        model.eval()
        total_eval_loss = 0
        count_eval = 0
        y_val_label_true = []
        y_val_label_predict = []
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            predict = model(x)
            y_val_label_true += list(y.squeeze().cpu().numpy())
            y_val_label_predict += list(predict[0].squeeze().cpu().detach().numpy())

            loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
            loss = loss_1
            total_eval_loss += float(loss)
            count_eval += 1
        
        auc = roc_auc_score(y_val_label_true, y_val_label_predict)
        print("Epoch %d val loss is %.3f, auc is %.3f " % (i + 1, total_eval_loss / count_eval, auc))

        # early stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
                state = model.state_dict()
                torch.save(state, model_path)
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break    


def train_multi_task(model, train_loader, val_loader, model_path, args):
    device = args.device
    epoch = args.epochs
    early_stop = 5
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)

    patience, eval_loss = 0, 0
    # train
    for i in range(epoch):
        model.train()
        y_train_click_true = []
        y_train_click_predict = []
        y_train_like_true = []
        y_train_like_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2) in enumerate(train_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_train_click_true += list(y1.squeeze().cpu().numpy())
            y_train_like_true += list(y2.squeeze().cpu().numpy())
            y_train_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        
        click_auc = roc_auc_score(y_train_click_true, y_train_click_predict)
        like_auc = roc_auc_score(y_train_like_true, y_train_like_predict)
        print("Epoch %d train loss is %.3f, click auc is %.3f and like auc is %.3f" % (i + 1, total_loss / count,
                                                                                            click_auc, like_auc))
        # validation
        model.eval()
        total_eval_loss = 0
        count_eval = 0
        y_val_click_true = []
        y_val_like_true = []
        y_val_click_predict = []
        y_val_like_predict = []
        for idx, (x, y1, y2) in enumerate(val_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_val_click_true += list(y1.squeeze().cpu().numpy())
            y_val_like_true += list(y2.squeeze().cpu().numpy())
            y_val_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_eval_loss += float(loss)
            count_eval += 1
        
        click_auc = roc_auc_score(y_val_click_true, y_val_click_predict)
        like_auc = roc_auc_score(y_val_like_true, y_val_like_predict)
        print("Epoch %d val loss is %.3f, click auc is %.3f and like auc is %.3f" % (i + 1,
                                                                                    total_eval_loss / count_eval,
                                                                                    click_auc, like_auc))

        # early stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
                state = model.state_dict()
                torch.save(state, model_path)
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break


def test_single_task(model, test_loader, args):
    device = args.device
    epochs = args.epoch
    loss_function = nn.BCEWithLogitsLoss()
    model.to(device)

    model.eval()
    total_test_loss = 0
    count_eval = 0
    y_test_label_true = []
    y_test_label_predict = []
    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        predict = model(x)
        y_test_label_true += list(y.squeeze().cpu().numpy())
        y_test_label_predict += list(predict[0].squeeze().cpu().detach().numpy())

        loss = loss_function(predict[0], y.unsqueeze(1).float())
        total_test_loss += float(loss)
        count_eval += 1

    auc = roc_auc_score(y_test_label_true, y_test_label_predict)
    print("Epoch %d test loss is %.3f, auc is %.3f" % (epochs, total_test_loss / count_eval, auc))


def test_multi_task(model, test_loader, args):
    device = args.device
    epochs = args.epoch
    loss_function = nn.BCEWithLogitsLoss()
    model.to(device)

    model.eval()
    total_test_loss = 0
    count_eval = 0
    y_test_click_true = []
    y_test_like_true = []
    y_test_click_predict = []
    y_test_like_predict = []
    for idx, (x, y1, y2) in enumerate(test_loader):
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        predict = model(x)
        y_test_click_true += list(y1.squeeze().cpu().numpy())
        y_test_like_true += list(y2.squeeze().cpu().numpy())
        y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
        y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

        loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
        loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
        loss = loss_1 + loss_2
        total_test_loss += float(loss)
        count_eval += 1
    
    click_auc = roc_auc_score(y_test_click_true, y_test_click_predict)
    like_auc = roc_auc_score(y_test_like_true, y_test_like_predict)
    print("Epoch %d test loss is %.3f, click auc is %.3f and like auc is %.3f" % (epochs + 1,
                                                                                    total_test_loss / count_eval,
                                                                                    click_auc, like_auc))


def mtlTrain(model, train_loader, val_loader, test_loader, args, train=True):
    model_path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if train:
        if args.mtl_task_num == 2:
            train_multi_task(model, train_loader, val_loader, model_path, args)
        else:
            train_single_task(model, train_loader, val_loader, model_path, args)
    else:
        state = torch.load(model_path)
        model.load_state_dict(state)
        if args.mtl_task_num == 2:
            test_multi_task(model, test_loader, args)
        else:
            test_single_task(model, test_loader, args)

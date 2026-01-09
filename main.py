import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
#from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch.nn.functional as F

from models import resnet,resnet_attribute_op
from models.fishr import compute_fishr_penalty
from models import resnet_mixstyle
from utils.dataloader import DATASET
from utils.fix_seeds import fix_random_seeds

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('MPCCI', add_help=False)

    # Model params

    parser.add_argument('--img_size',type=int,default=128)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # initial 0
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset',default='ct',type=str)   # COVID ot BreakHis
    parser.add_argument('--root', type=str, default='./data')

    parser.add_argument('--num_classes',type=int)
    parser.add_argument('--job_id',default=0)
    parser.add_argument('--model',default='resnet_attribute',type=str)
    parser.add_argument('--num_att',default=5,type=int)
    parser.add_argument('--mixup_alpha', default=0.4, type=float, help='mixup alpha (Beta distribution)')
    parser.add_argument('--mixup_prob', default=1.0, type=float, help='probability to apply mixup')

    return parser
def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def cross_entropy_index(hat_y,y,different_y,num_classes,ex_index,ratio,args):
    label = torch.tensor(np.eye(num_classes, dtype=np.uint8)[y.cpu().numpy()]).to(y.device).float()
    different_label=torch.tensor(np.eye(num_classes, dtype=np.uint8)[different_y.cpu().numpy()]).to(y.device).float()

    exp_hat_y = torch.exp(hat_y)

    upper_loss_cal = torch.sum(label * exp_hat_y, dim=1, keepdim=True)
    below_loss_cal = torch.sum(exp_hat_y, dim=1, keepdim=True)

    upper_loss_cal_different=torch.sum(different_label * exp_hat_y, dim=1, keepdim=True)

    loss = -torch.log(upper_loss_cal / below_loss_cal)
    loss_different=-torch.log(upper_loss_cal_different/below_loss_cal)

    ex_index=ex_index.view(ex_index.shape[0],args.num_att,-1)
    ex_index=ex_index[:,:,0]
    ex_index=torch.sum(ex_index,dim=1)
    ex_index=ex_index/args.num_att
    ratio1=torch.min(torch.ones_like(ex_index),ex_index+ratio)
    ratio2=torch.max(torch.zeros_like(ex_index),1-ex_index-ratio)
    loss=loss*ratio1+ratio2*loss_different
    return loss.mean()

def main(args):
    fix_random_seeds(args.seed)
    device = torch.device(args.device)
    cudnn.benchmark = True
    if args.dataset == 'TN5000':
        args.num_classes = 2
        train_dir = './data/TN5000/train.txt'
        val_dir = './data/TN5000/val.txt'
        test_dir = './data/TN5000/test.txt'
    elif args.dataset == 'tn3k':
        args.num_classes = 2
        train_dir = './data/tn3k/train.txt'
        val_dir   = './data/tn3k/val.txt'
        test_dir  = './data/tn3k/test.txt'


    print(args)
    dataset_val = DATASET(args.dataset, val_dir, args.root, args.img_size, is_train=False)
    print(f"Validation data loaded: there are {len(dataset_val)} images.")

    dataset_train = DATASET(args.dataset, train_dir, args.root, args.img_size)
    print(f"Train data loaded: there are {len(dataset_train)} images.")

    dataset_test = DATASET(args.dataset, test_dir, args.root, args.img_size, is_train=False)
    print(f"Test data loaded: there are {len(dataset_test)} images.")


    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, drop_last=False)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.num_workers, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

    if args.model in ['resnet', 'resnet_fishr', 'resnet_mixup']:
        model = resnet.Model(args.num_classes)
    elif args.model == 'resnet_mixstyle':
        model = resnet_mixstyle.Model(args.num_classes, pretrained=True, p=0.5, alpha=0.1)
    else:
        model=resnet_attribute_op.Model(args.num_classes,num_att=args.num_att)
    model.to(device)
    # class imbalance handling for TN3K
   
    criterion = nn.CrossEntropyLoss()

   
    parameters = model.parameters()

    if args.dataset in ['TN5000', 'tn3k']:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)  ## 部署在GPU  todo
    num_train=len(dataset_train)
    num_test=len(dataset_test)
    num_val=len(dataset_val)

    train_prec=[]
    max_val_epoch=0.0

    max_val_array=np.array([0.0])
    max_test_array=np.array([0.0])

    train_att_bank=torch.zeros(num_train,512*(args.num_att+1)).to(device)
    train_label_bank=torch.zeros(num_train).long().to(device)
    attribute_prototype=torch.rand(args.num_classes,512*(args.num_att+1)).to(device)
    # attribute=torch.zeros(args.num_classes,2048*5).to(device)

    for epoch in range(args.epoch):
        if epoch>1000:
            break
        else:
            model.train()
            loss = 0.0
            prec = 0.0
            for i, (index, img, label, _) in enumerate(data_loader_train):
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                # same category attribute prototype
                same_category_att = attribute_prototype[label, :]

                # different category attribute prototype
                randind = (torch.rand(img.shape[0]) * (args.num_classes - 1)).long() + 1
                randind = (randind.to(device) + label) % (args.num_classes)
                different_category_att = attribute_prototype[randind, :]
                if args.model in ['resnet', 'resnet_fishr', 'resnet_mixstyle', 'resnet_mixup']:

                   raw_logits = model(img)

                   x = None
                   mixed_y = None
                   mixed_index = None
                else:

                   x, raw_logits, mixed_y, mixed_index = model(img, same_category_att, different_category_att, label)

                if args.model == 'resnet_mixup' and np.random.rand() < args.mixup_prob:
                   mixed_img, y_a, y_b, lam = mixup_data(img, label, alpha=args.mixup_alpha)
                   raw_logits = model(mixed_img)
                   total_loss = mixup_loss(criterion, raw_logits, y_a, y_b, lam)


                else:
                   raw_loss = criterion(raw_logits, label)

               
                   if args.model == 'resnet_fishr':
                      fishr_penalty = compute_fishr_penalty(
                      logits=raw_logits,
                      labels=label,
                      classifier=model.classifier,
                      num_envs=2
                     )
                      lambda_fishr = 1.0
                      total_loss = raw_loss + lambda_fishr * fishr_penalty
                   elif args.model == 'resnet_attribute' and epoch > 2:
                    raw_loss = raw_loss + 0.1 * cross_entropy_index(mixed_y, label, randind, args.num_classes, mixed_index,
                                                                    torch.min(torch.tensor([1.0, (epoch + 100) / args.epoch]).to(device)),
                                                                    args)  # todo torch.min(torch.tensor([1.0, (epoch + 100) / args.epoch])

                    total_loss = raw_loss
                   else:
                    total_loss = raw_loss
                loss = loss + total_loss
                if args.model== 'resnet_attribute':
                   train_att_bank[index, :] = x.data
                # print(label)
                train_label_bank[index] = label
                total_loss.backward()

                optimizer.step()

                pred = raw_logits.max(1, keepdim=True)[1]
                prec += pred.eq(label.view_as(pred)).sum().item()

            print('[EPOCH:{}],loss:{:.4f},prec:{:.4f}'.format(epoch, loss, prec / num_train))
            train_prec.append(prec / num_train)
            if args.model == 'resnet_attribute':
              train_label_bank_oh = torch.tensor(
               np.eye(args.num_classes, dtype=np.uint8)[train_label_bank.cpu().numpy()]
              ).to(device).float()
    

              attribute_prototype = (train_label_bank_oh.T).matmul(train_att_bank) / (
                torch.sum(train_label_bank_oh.T, dim=1, keepdim=True)
              )


            if epoch % 1 == 0:
                model.eval()
                val_label_bank = torch.zeros(num_val, args.num_classes).to(device)
                val_GT_bank = torch.zeros(num_val).to(device).long()

                test_label_bank = torch.zeros(num_test, args.num_classes).to(device)
                test_GT_bank = torch.zeros(num_test).to(device).long()

                with torch.no_grad():

                    val_loss = 0.0
                    val_prec = 0.0
                    for i, (index, img, label, _) in enumerate(data_loader_val):
                       img = img.to(device)
                       label = label.to(device)

                       if args.model in ['resnet', 'resnet_fishr', 'resnet_mixstyle', 'resnet_mixup']:
                           raw_logits = model(img)
                           x = None
                           mixed_y = None
                           mixed_index = None
                       else:
                         _, raw_logits = model(img)   # ⭐ 只传 img

                        #_, raw_logits = model(img)
                       raw_loss = criterion(raw_logits, label)
                       total_loss = raw_loss
                       val_loss = val_loss + total_loss

                       pred = raw_logits.max(1, keepdim=True)[1]
                       val_prec += pred.eq(label.view_as(pred)).sum().item()
                       val_label_bank[index] = raw_logits
                       val_GT_bank[index] = label

                    val_acc, val_precision, val_recall, val_f1 = eval_tfpn(val_label_bank, val_GT_bank, args)

                    test_loss = 0.0
                    test_prec = 0.0
                    for i, (index, img, label, _) in enumerate(data_loader_test):
                       img = img.to(device)
                       label = label.to(device)

                       if args.model in ['resnet', 'resnet_fishr', 'resnet_mixstyle', 'resnet_mixup']:
                         raw_logits = model(img)
                         x = None
                         mixed_y = None
                         mixed_index = None
                       else:
                         _, raw_logits = model(img)

                       raw_loss = criterion(raw_logits, label)
                       total_loss = raw_loss
                       test_loss = test_loss + total_loss

                       pred = raw_logits.max(1, keepdim=True)[1]
                       test_prec += pred.eq(label.view_as(pred)).sum().item()

                       test_label_bank[index] = raw_logits
                       test_GT_bank[index] = label

                    test_acc, test_precision, test_recall, test_f1 = eval_tfpn(test_label_bank, test_GT_bank, args)
                    if (val_acc+val_recall+val_precision+val_f1)>max_val_epoch:
                        max_val_epoch=val_acc+val_recall+val_precision+val_f1
                        max_val_array=np.array([val_acc,val_precision,val_recall,val_f1])
                        max_test_array=np.array([test_acc,test_precision,test_recall,test_f1])
                    print('epoch:{},val_acc:{:.4f},val_prec:{},val_recall:{},val_F1:{}'.format(epoch, val_acc,val_precision,val_recall,val_f1))
                    print('epoch:{},test_acc:{:.4f},test_prec:{},test_recall:{},test_F1:{}'.format(epoch, test_acc,
                                                                                               test_precision,
                                                                                               test_recall, test_f1))
                    print('Max validation    : ', max_val_array)
                    print('Corresponding test: ', max_test_array)


def eval_tfpn(output, target, args):
    output=torch.softmax(output,dim=1)
    output=torch.max(output,dim=1)[1]
    if args.dataset=='breastPublic':
        output=1-output
        target=1-target
    output=output.cpu().numpy()
    target=target.cpu().numpy()

    tp = np.sum(output*target)
    fp = np.sum(output*(1-target))
    fn = np.sum((1-output)*target)
    tn = np.sum((1-output)*(1-target))

    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 =2*precision*recall/(precision+recall)

    return acc, precision, recall, F1

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Training time in: %s' %((end_train - start_train)/3600))

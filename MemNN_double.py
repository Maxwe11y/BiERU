#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
import torch.nn.functional as F

import argparse

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1) # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class RNTN(nn.Module):

    def __init__(self, input_dim, n_class, l):
        super(RNTN, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.window = 5
        self.V = nn.Parameter(torch.zeros((input_dim, 1, 2*input_dim, 2*input_dim)))
        #self.UU = nn.Parameter(torch.zeros((input_dim, 1, 2*input_dim, 10)))
        #self.VV = nn.Parameter(torch.zeros((input_dim, 1, 10, 2*input_dim)))
        #self.diag = nn.Parameter(torch.zeros((1, 1, 2*input_dim, 2*input_dim)))
        #self.V_s = nn.Parameter(torch.zeros((input_dim, 1, input_dim, input_dim))).cuda()

        self.W = nn.Linear(2*input_dim, input_dim)
        
        self.Ws = nn.Linear(2*(52 + input_dim), n_class)
        #self.We = nn.Linear(2*input_dim, n_class)
        self.gru = nn.LSTMCell(input_size=input_dim, hidden_size=input_dim)
        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.l = l
        self.cnn3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=4, stride=2)
        self.dropout = nn.Dropout(0.8)
        # self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, dropout=0.1, bidirectional=True)
        if l:
            self.L = nn.Linear(2*input_dim, input_dim)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, mask):
        """
        :param U:-->seq, batch, dim
        :return:
        """
        v_mask = torch.rand(self.V.size())
        v_mask = torch.where(v_mask > 0.15, torch.full_like(v_mask, 1), torch.full_like(v_mask, 0)).cuda()
        self.V = nn.Parameter(self.V * v_mask)

        results1 = torch.zeros(0).type(U.type())
        results2 = torch.zeros(0).type(U.type())
        h = torch.zeros((U.size(1), U.size(2))).cuda()
        c = torch.zeros((U.size(1), U.size(2))).cuda()

        for i in range(U.size()[0]):
            if i == 0:
                #p = U[i]
                #t_t = U[i-1]
                t_t = U[i]
                v_cat = torch.cat((t_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))
                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
                h_cat = torch.cat((h, h3), dim=1)

                h_cat = self.dropout(h_cat)
                results1 = torch.cat((results1, h_cat))

            else:
                
                l_t = U[i-1]
                #l_t = p
                t_t = U[i]
                
                v_cat = torch.cat((l_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                #p = self.ac_linear(p)
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))
                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
                h_cat = torch.cat((h, h3), dim=1)
                h_cat = self.dropout(h_cat)
                results1 = torch.cat((results1, h_cat))
                
        
        rever_U = self._reverse_seq(U, mask)

        for i in range(rever_U.size()[0]):
            # get temp and last, (batch, dim)
            if i == 0:
                #p = rever_U[i]
                #t_t = rever_U[i-1]
                t_t = rever_U[i]
                v_cat = torch.cat((t_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))

                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
                h_cat = torch.cat((h, h3), dim=1)
                h_cat = self.dropout(h_cat)
                results2 = torch.cat((results2, h_cat))
            else:
                
                l_t = rever_U[i-1]
                #l_t = p
                t_t = rever_U[i]
                v_cat = torch.cat((l_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(
                    m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                #p = self.ac_linear(p)
                p = self.dropout(p)
                # h = self.gru(p, h)
                h, c = self.gru(p, (h, c))


                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
                h_cat = torch.cat((h, h3), dim=1)

                h_cat = self.dropout(h_cat)
                results2 = torch.cat((results2, h_cat))

        results2 = results2.contiguous().view(rever_U.size(0), rever_U.size(1), -1)
        results2 = self._reverse_seq(results2, mask)
        results2 = results2.contiguous().view(results1.size(0), results1.size(1))

        #results = torch.log_softmax(self.Ws(results1), dim=1)
        results = torch.log_softmax(self.Ws(torch.cat((results1, results2), dim=1)), dim=1)
        # results = torch.log_softmax(self.Ws(torch.cat((results1, results2, bioutput), dim=1)), dim=1)

        return results


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    #it = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes
        log_prob = model(textf, umask)  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(log_prob, labels_, umask)

        #if train and it % 10 == 0:
        #    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model,
        #                                                                                            loss_function,
        #                                                                                            test_loader, e)
        #    print(test_acc)
        pred_ = torch.argmax(log_prob, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            # print(torch.mean(model.V.grad))
            optimizer.step()
        #it += 1
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    batch_size = args.batch_size
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs

    D_m = 100

    model = RNTN(D_m, n_classes, False)
    print('\n number of parameters {}'.format(sum([p.numel() for p in model.parameters()])))
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.99)

    train_loader, valid_loader, test_loader =\
            get_IEMOCAP_loaders(r'./IEMOCAP_features_raw.pkl',
                                valid=0.0,
                                batch_size=batch_size,
                                num_workers=2)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_, train_fscore = train_or_eval_model(model, loss_function,
                                                train_loader, e, optimizer, True)
        valid_loss, valid_acc, _,_,_, val_fscore = train_or_eval_model(model, loss_function, valid_loader, e)
        #scheduler.step()
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask =\
                    test_loss, test_label, test_pred, test_mask

        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))



    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss,
                                     round(accuracy_score(best_label, best_pred, sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    # with open('best_attention.p','wb') as f:
    #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)

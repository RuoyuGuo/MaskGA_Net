import json
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import cv2 as cv

from utils import my_dataset, train_kit
from utils.img_utils import to_npimg
from network import SegNet, SegNet_cons, AttNet, PseudoEdgeNet
from loss import BCE_loss, L1_loss
from metrics import IoU_eval, Dice_eval


def cal_metrics(metrics, pred, yb):
    metrics_record = {}

    for m_name, m_fun in metrics.items():
        metrics_record[m_name] = m_fun(pred, yb[:,0]).item()

    return metrics_record

def cal_batch(Kit, model_set, xb, yb, opt=None):
    '''
    model set index:
    0: SegNet or SegNet_cons
    1: Attention net

    supervision index:
    0: ground truth
    1: voronoi + point
    2: psuedo labels
    -2: revised pseudo labels
    -1: confusion metircs (boolean, denote if label is noisy)
    '''
    #define segnet supervision index,  attnet supervision index
    #attention net input,
    if Kit.cfgs.use_cl is True:
        ss_index = -2
        attnet_input = -2
    elif Kit.cfgs.full_sup is True:
        ss_index = 0
        attnet_input = 2
    else:
        ss_index = 2
        attnet_input = 2
    
    as_index = 1

    #define network output
    #get output

    if Kit.cfgs.framework == 'Seg':
        y_seg = model_set[0](xb)
        
    if Kit.cfgs.framework == 'Seg_Att':
        y_seg = model_set[0](xb)
        y_att = model_set[1](torch.cat((xb, yb[:,[attnet_input]]), dim=1))

    if Kit.cfgs.framework == 'SAC':
        y_seg, y_cons, _  = model_set[0](xb)
        y_att  = model_set[1](torch.cat((xb, yb[:,[attnet_input]]), dim=1))

    if Kit.cfgs.framework == 'PseudoEdgeNet':
        y_seg, y_edge, y_grad = model_set[0](xb)
        
    
    #calculate loss
    #{loss_name: [value, weight]}
    loss_record_grad = {}
    loss_record_item = {}
    for l_name, l_fun in Kit.loss.items():
        if l_name == 'seg_loss':
            if Kit.cfgs.framework in ['Seg', 'PseudoEdgeNet']:
                loss_record_grad[l_name] = [l_fun(y_seg, yb[:,ss_index]), \
                                                    Kit.cfgs.seglossw]
            else:
                loss_record_grad[l_name] = [l_fun(y_seg*y_att, yb[:,ss_index]), \
                                                    Kit.cfgs.seglossw]

        if l_name == 'att_loss':
            loss_record_grad[l_name] = [l_fun(y_att, yb[:,as_index]), \
                                                    Kit.cfgs.attlossw]
        
        if l_name == 'cons_loss':
            loss_record_grad[l_name] = [l_fun(y_cons, y_seg), Kit.cfgs.conslossw]

        if l_name == 'edge_loss':
            loss_record_grad[l_name] = [l_fun(y_edge, y_grad), 1]

    #sum and backward, 0 is loss value, 1 is loss weight
    loss_record_grad['total_loss'] = 0
    for key, value in loss_record_grad.items():
        if key != 'total_loss':
            loss_record_grad['total_loss'] += value[0] * value[1]

    #record value
    for key, value in loss_record_grad.items():
        if key != 'total_loss':
            loss_record_item[key] = value[0].item()
        else:
            loss_record_item[key] = value.item()

    #training phase
    if opt is not None:
        opt.zero_grad()    
        loss_record_grad['total_loss'].backward()
        opt.step()
   
    #calculate metrics
    metrics = cal_metrics(Kit.metrics, y_seg, yb) 


    return loss_record_item, metrics

def train(cfgs):
    random.seed(cfgs.seed)
    
    #init training kit
    Kit = train_kit.Kit(cfgs)
    Kit.log_configs()
    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    #load dataset
    #train_dl, val_dl, test_dl, test_ds, len_train, len_val, _, _ =\
    #my_dataset.get_data(**Kit.get_paras_for_dataset())
    train_dl, val_dl, test_dl, test_ds = my_dataset.get_data(Kit.get_paras_for_dataset())

    #load network
    model_set = []

    if cfgs.stage == 0 or cfgs.mode_stage2 == 'retrain':
        print('Constructing Network...')
        if cfgs.framework == 'Seg':
            mySegNet = SegNet.build_network(device, 32, cfgs.is_1000)
            model_set.append(mySegNet)
        if cfgs.framework == 'Seg_Att':
            mySegNet = SegNet.build_network(device, 32, cfgs.is_1000)
            myAttNet = AttNet.build_network(device, 32, cfgs.is_1000) 
            model_set.append(mySegNet)
            model_set.append(myAttNet)
        if cfgs.framework == 'SAC':
            mySegNet = SegNet_cons.build_network(device, 32, cfgs.cons_net_n, cfgs.is_1000)
            myAttNet = AttNet.build_network(device, 32, cfgs.is_1000)
            model_set.append(mySegNet)
            model_set.append(myAttNet)
        if cfgs.framework == 'PseudoEdgeNet':
            mySegNet = PseudoEdgeNet.build_network(device, cfgs.is_1000)        
            model_set.append(mySegNet)

    elif cfgs.mode_stage2 == 'refine':
        print('Loading pretrained Network...')
        model_set = torch.load(Kit.network_PATH, map_location=device)

    #move network
    for e in model_set:
        e.to(device)

    #define optimizer
    paras_set = []
    for e in model_set:
        paras_set += list(e.parameters())

    optimizer = torch.optim.Adam(paras_set,
                                lr=cfgs.my_lr, 
                                weight_decay=cfgs.weight_decay)

    #define rate schedule   
    BEST_MONITOR = float('inf')
    BEST_EPOCH = 0                 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, \
                                                cfgs.my_patience, \
                                                verbose=True)


    print('Training start...')
    print('')



    for epoch in range(cfgs.epochs):
        #print('training_loop.py:')
        #print(torch.cuda.max_memory_allocated())

        #training phase
        SINCE = time.time()
        
        for e in model_set:
            e.train()

        #update per batch
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, metrics = cal_batch(Kit, model_set, xb, yb, opt = optimizer)

            #create stats
            stats = {
                'update'    : 'batch',
                'phase'     : 'train',
                'loss'      : loss,
                'metrics'   : metrics,
                'size'      : len(xb)
                }

            Kit.log(stats)

        #update per epoch 
        stats = {
            'update'    : 'epoch',
            'time'      : time.time() - SINCE,
            'phase'     : 'train',
            'epoch'     : epoch
        }
        Kit.log(stats)


        #Validation
        for e in model_set:
            e.eval()             #remove dropout, bn...

        with torch.no_grad():    #stop calculate gradients
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss, metrics = cal_batch(Kit, model_set, xb, yb)

                #create stats
                stats = {
                    'update'    : 'batch',
                    'phase'     : 'val',
                    'loss'      : loss,
                    'metrics'   : metrics,
                    'size'      : len(xb)
                    }

                Kit.log(stats)

            #Update of total validation set, and save best result
            stats = {
                'update'    : 'epoch',
                'phase'     : 'val'
                }
            Kit.log(stats)

            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss, metrics = cal_batch(Kit, model_set, xb, yb)
                
                #create stats
                stats = {
                    'update'    : 'batch',
                    'phase'     : 'test',
                    'loss'      : loss,
                    'metrics'   : metrics,
                    'size'      : len(xb)
                    }

                Kit.log(stats)

            #Update of total test set
            stats = {
                'update'    : 'epoch',
                'phase'     : 'test'
                }
            Kit.log(stats)

        #if need to save best
        last_total_loss = Kit.get_last_loss('val')
        if last_total_loss < BEST_MONITOR:
            BEST_MONITOR = last_total_loss
            BEST_EPOCH = epoch
            Kit.save_best(model_set, BEST_MONITOR)
        
        scheduler.step(last_total_loss)

    Kit.save_final(model_set)
    Kit.eval(model_set, test_ds)
    
    
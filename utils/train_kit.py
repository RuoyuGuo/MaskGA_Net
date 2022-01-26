'''
Training logger and toolkit used for MaskGA_Net
'''

import json
import os

import torch
import numpy as np
from matplotlib import pyplot as plt
from dotmap import DotMap

from loss import BCE_loss, L1_loss
from metrics import Dice_eval, IoU_eval
from utils.img_utils import to_npimg

def format_time(s):
    s = round(s)

    #s
    if s < 60:
        return f'{s}s'
    #m s
    elif s < 60 * 60:
        return f'{s // 60:02}m {s % 60:02}s'
    #h m s
    elif s < 60 * 60 * 24:
        return f'{s // (60*60):02}h {(s // 60) % 60 :02}m {s % 60:02}s'
    #d h m
    else:
        return f'{s // (24 * 60 * 60)}d {(s // (60 * 60)) % 24:02}h {(s // 60) % 60}m'

class Kit():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.cp_PATH = self._init_cp_path()
        self.network_PATH = self._init_network_path()
        self.log_PATH = self._init_log()

        self.loss = self._init_loss()
        self.metrics = self._init_metrics()

        self.loss_epochs = {'train':[], 'val':[], 'test':[], 'time': 0.0}
        self.metrics_epochs = {'train':[], 'val':[], 'test':[]}

        self.loss_batch = {loss_name:0.0 for loss_name in self.loss}
        self.metrics_batch = {metrics_name:0.0 for metrics_name in self.metrics}
        self.size_batch = 0


    def _init_log(self):
        log_PATH = os.path.join(self.cp_PATH, 'log.txt')

        with open(log_PATH, 'w') as f:
            line = []
            line += ['Initialising logs...']
            line += [f'Training for {self.cfgs.epochs} epochs']
            line += [f'Dataset: {self.cfgs.dataset}']
            line += [f'Framework: {self.cfgs.framework}']
            line += [f'If use CL: {self.cfgs.use_cl}, Iteration times: {self.cfgs.stage}']

            print('\n'.join(line))
            print()
            f.write('\n'.join(line))
            f.write('\n')
            f.write('\n')

        return log_PATH

    def _init_cp_path(self):
        #log path
        cp_PATH = os.path.join('.', 'checkpoints', self.cfgs.dataset, 
                        'stage'+str(self.cfgs.stage),  self.cfgs.model_name)

        os.makedirs(cp_PATH, exist_ok=True)

        return cp_PATH


    def _init_network_path(self):
        #pre network path if 
        if self.cfgs.stage > 0:
            if self.cfgs.cp_time == 'best':
                network_PATH = os.path.join('.', 'checkpoints', self.cfgs.dataset, \
                                                'stage'+str(self.cfgs.stage-1),  \
                                                self.cfgs.model_name , str(self.cfgs.k_fold)+'.pt')
            else:
                network_PATH = os.path.join('.', 'checkpoints',  self.cfgs.dataset, \
                                                'stage'+str(self.cfgs.stage-1), \
                                                self.cfgs.model_name, str(self.cfgs.k_fold)+'_f.pt')
        else:
            network_PATH = 'nopre'

        return network_PATH


    def _init_loss(self):
        with open(self.log_PATH, 'a') as f:
            line = f'Initialising training loss...'
            print(line)
            print()

            f.write(line)
            f.write('\n')
            f.write('\n')

        if self.cfgs.framework == 'Seg':
            loss = {'seg_loss': BCE_loss.full()}
        if self.cfgs.framework == 'Seg_Att':
            loss = {'seg_loss': BCE_loss.full(),
                        'att_loss': L1_loss.partial(w1=self.cfgs.attlossw_p, \
                                                    w2=self.cfgs.attlossw_v)}
        if self.cfgs.framework == 'SAC':
            loss = {'seg_loss': BCE_loss.full(),
                        'att_loss': L1_loss.partial(w1=self.cfgs.attlossw_p, \
                                                    w2=self.cfgs.attlossw_v),
                        'cons_loss': L1_loss.vanilla()}
        if self.cfgs.framework == 'PseudoEdgeNet':
            if self.cfgs.full_sup is True:
                loss = {'seg_loss': BCE_loss.full(),
                            'edge_loss': L1_loss.vanilla()}
            else:
                loss = {'seg_loss': BCE_loss.partial(1, 0.1),
                            'edge_loss': L1_loss.vanilla()}

        loss['total_loss'] = None

        return loss

    def _init_metrics(self):
        with open(self.log_PATH, 'a') as f:
            line = f'Initialising evaluation metrics...'
            print(line)
            print()

            f.write(line)
            f.write('\n')
            f.write('\n')

        metrics = {'IoU': IoU_eval.sigmoid(),
                    'Dice': Dice_eval.sigmoid()}

        return metrics

    def log_configs(self):
        '''
        show configuratinos
        ''' 

        formatted_configs = {
            'Dataset':{
                'dataset'           : self.cfgs.dataset,
                'k_fold'            : self.cfgs.k_fold,
                'random_shift'      : self.cfgs.random_shift, 
                'mode_label'        : self.cfgs.mode_label,
                'cut_size'          : self.cfgs.cut_size,
                'data_norm'         : self.cfgs.data_norm, 
                'seed'              : self.cfgs.seed,
                'color_norm'        : self.cfgs.color_norm
            },
            'Framework':{
                'stage'             : self.cfgs.stage,
                'model_name'        : self.cfgs.model_name,
                'framework'         : self.cfgs.framework, 
                'use_cl'            : self.cfgs.use_cl,
                'full_sup'          : self.cfgs.full_sup,
                'run_pipe'          : self.cfgs.run_pipe,       
                'mode_stage2'       : self.cfgs.mode_stage2
            },
            'Network':{
                'cons_net_n'        : self.cfgs.cons_net_n,
                'is_1000'           : self.cfgs.is_1000
            },
            'Training':{
                'cp_time'           : self.cfgs.cp_time,
                'bs'                : self.cfgs.bs,
                'device'            : self.cfgs.device,
                'epochs'            : self.cfgs.epochs,
                'attlossw_p'        : self.cfgs.attlossw_p,      
                'attlossw_v'        : self.cfgs.attlossw_v,    
                'seglossw'          : self.cfgs.seglossw,       
                'attlossw'          : self.cfgs.attlossw,     
                'conslossw'         : self.cfgs.conslossw,    
                'my_lr'             : self.cfgs.my_lr,
                'weight_decay'      : self.cfgs.weight_decay,
                'my_patience'       : self.cfgs.my_patience,
            },
            'Checkpoint_path'       : self.cp_PATH  
        }     

        #create and save config
        print(f'Network, log files, results are saved in {self.cp_PATH}')
        print()
        os.makedirs(self.cp_PATH, exist_ok=True)
        with open(os.path.join(self.cp_PATH, 'configs.json'), 'w') as f:
            json.dump(formatted_configs, f, indent=2)

        #display options
        print(f'Display configurations:')
        print(json.dumps(formatted_configs, indent=2))
        print()


    def log(self, stats):
        '''
        stats can have these :
        {'phase': 'train',  'update': 'batch', 
        'size': value,      'epoch': value , 
        'loss': {'loss_name': value, 'loss_name2': value},
        'metrics': {'metrics_name': value, 'metrics_name2': value}
        'time': value(in seconds)
        }
        '''
        #log per batch
        if stats['update'] == 'batch':
            assert stats['phase'] in ['train', 'val', 'test']
            assert stats['loss'].keys() == self.loss_batch.keys()
            assert stats['metrics'].keys() == self.metrics_batch.keys()

            #log loss
            for loss_name, loss_value in stats['loss'].items():
                self.loss_batch[loss_name] += loss_value * stats['size']

            #log metrics
            for metric_name, metric_value in stats['metrics'].items():
                self.metrics_batch[metric_name] += metric_value * stats['size']

            self.size_batch += stats['size']

        #log per epoch
        if stats['update'] == 'epoch':
            #average loss, metrics
            for key, value in self.loss_batch.items():
                self.loss_batch[key] = value/self.size_batch

            for key, value in self.metrics_batch.items():
                self.metrics_batch[key] = value/self.size_batch

            if stats['phase'] == 'train':
                self.loss_epochs['time'] += int(stats["time"])
                title = []
                title += [f'Epoch {stats["epoch"]:<13} Training time/sec {round(stats["time"]):<7d}']
                title += [f'Total time {format_time(self.loss_epochs["time"])}']
                line = []
                line += [f'Training result: '+' '*2] 
   
            if stats['phase'] == 'val':
                line = []
                line += [f'Validation result: ']
            if stats['phase'] == 'test':
                line = []
                line += [f'Test result: '+' '*6]

            line += [f'loss:']
            line += [' '.join(f'{key} {value:<.4f}' for key,value in self.loss_batch.items())]
            line += [f'Metrics:']
            line += [' '.join(f'{key} {value:<.4f}' for key,value in self.metrics_batch.items())]

            if stats['phase'] == 'train': 
                print(' '.join(title))
                print(' '.join(line))
                with open(self.log_PATH, 'a') as f:
                    f.write(' '.join(title))
                    f.write('\n')
                    f.write(' '.join(line))
                    f.write('\n')

            else:
                print(' '.join(line))
                with open(self.log_PATH, 'a') as f:
                    f.write(' '.join(line))
                    f.write('\n')

            #append to total loss
            #initial loss, metrics per batch
            self.loss_epochs[stats['phase']].append(self.loss_batch)
            self.metrics_epochs[stats['phase']].append(self.metrics_batch)

            self.loss_batch = {loss_name:0.0 for loss_name in self.loss}
            self.metrics_batch = {metrics_name:0.0 for metrics_name in self.metrics}
            self.size_batch = 0


    def get_paras_for_dataset(self):
        data_import_options = {
            'stage'             : self.cfgs.stage,
            'cp_time'           : self.cfgs.cp_time,
            'dataset'           : self.cfgs.dataset,
            'k_fold'            : self.cfgs.k_fold,
            'random_shift'      : self.cfgs.random_shift, 
            'model_name'        : self.cfgs.model_name,
            'mode_label'        : self.cfgs.mode_label,
            'cut_size'          : self.cfgs.cut_size,
            'bs'                : self.cfgs.bs,
            'seed'              : self.cfgs.seed,
            'data_norm'         : self.cfgs.data_norm,
            'device'            : self.cfgs.device,
            'color_norm'        : self.cfgs.color_norm,
            'network_path'      : self.network_PATH
        }

        return DotMap(data_import_options, _dynamic=False)

    def get_last_loss(self, phase):
        assert phase in ['train', 'val', 'test']
        return self.loss_epochs[phase][-1]['total_loss']

    def save_best(self, model, BEST_MONITOR):
        #save network
        line = f'Save best model with loss: {BEST_MONITOR:<.4f}'         
        print(line)
        with open(self.log_PATH, 'a') as f:
            f.write(line)
            f.write('\n')

        torch.save(model, os.path.join(self.cp_PATH, str(self.cfgs.k_fold)+'.pt'))


    def save_final(self, model):

        #save final network
        line = []
        line += [f'Saving final model...']
        line += [f'Training end']
        print(' '.join(line))
        print()
        with open(self.log_PATH, 'a') as f:
            f.write(' '.join(line))
            f.write('\n')
            f.write('\n')

        torch.save(model, os.path.join(self.cp_PATH, str(self.cfgs.k_fold)+'_f.pt'))

        #save loss, metrics per epoch
        with open(os.path.join(self.cp_PATH, 'loss_log.json'), 'w') as f:
            json.dump(self.loss_epochs, f, indent=2)
        with open(os.path.join(self.cp_PATH, 'metrics_log.json'), 'w') as f:
            json.dump(self.metrics_epochs, f, indent=2) 

    def eval(self, model, test_ds):
        line = []
        line += [f'Evaluating on the test dataset:']
        print(' '.join(line))
        with open(self.log_PATH, 'a') as f:
            f.write(' '.join(line))
            f.write('\n')

        #evaluation on test dataset, and display network prediction
        if self.cfgs.cp_time == 'best':
            model = torch.load(os.path.join(self.cp_PATH, str(self.cfgs.k_fold)+'.pt'))

        if self.cfgs.cp_time == 'final':
            model = torch.load(os.path.join(self.cp_PATH, str(self.cfgs.k_fold)+'_f.pt'))
        
        for e in model:
            e.eval()
        
        network_output = []
        metrics_record = {metrics_name:[] for metrics_name in self.metrics}

        with torch.no_grad():     
            
            for xb, yb in test_ds:
                xb, yb = xb.to(self.cfgs.device).unsqueeze(0), yb.to(self.cfgs.device).unsqueeze(0)

                if self.cfgs.framework == 'Seg':
                    y_seg = model[0](xb)
                    network_output.append([[y_seg, 'f_seg']])

                if self.cfgs.framework == 'Seg_Att':
                    y_seg = model[0](xb)  
                    if self.cfgs.use_cl is True:
                        f_att = model[1](torch.cat((xb, yb[:,[-2]]), dim=1))
                    else:
                        f_att = model[1](torch.cat((xb, yb[:,[2]]), dim=1))

                    network_output.append([[y_seg, 'f_seg'], [f_att, 'f_att']])

                if self.cfgs.framework == 'SAC':
                    y_seg, _, f_cons = model[0](xb)

                    if self.cfgs.use_cl is True:
                        f_att = model[1](torch.cat((xb, yb[:,[-2]]), dim=1))
                    else:
                        f_att = model[1](torch.cat((xb, yb[:,[2]]), dim=1))

                    network_output.append([[y_seg, 'f_seg'], [f_att, 'f_att'], [f_cons[:,0], 'f_cons']])

                if self.cfgs.framework == 'PseudoEdgeNet':
                    y_seg, y_edge, _ = model[0](xb)
                    network_output.append([[y_seg, 'f_seg'], [y_edge, 'f_edge']])

                for m_name, m_fun in self.metrics.items():
                    metrics_record[m_name].append(m_fun(y_seg, yb[:, 0]).item())
        
        line = []
        line += [' '.join(f'{key:<13}'for key in metrics_record.keys())]
        for row in np.array(list(metrics_record.values())).T:
            line += [' '.join(f'{element:<13.4f}' for element in row)]
        line += ['MEAN and STD:']
        line += [' '.join(f'{np.mean(value):<.4f}({np.std(value):<.3f})' for key, value in metrics_record.items())]

        print('\n'.join(line))
        with open(self.log_PATH, 'a') as f:
            f.write('\n'.join(line))
            f.write('\n')
            f.write('\n')
        
        #plot
        fig, axs = plt.subplots(len(test_ds), len(network_output[0])+2, \
                                figsize=(5*(len(network_output[0])+2), 5*len(test_ds)))
        
        for i in range(len(test_ds)):
            for j in range(len(network_output[0])+2):
                #Input image
                if j == 0:
                    axs[i][j].imshow(to_npimg(test_ds[i][0]))
                    axs[i][j].axis('off')
                    axs[i][j].set_title('Input image')

                #Ground truth  
                elif j == 1:
                    axs[i][j].imshow(to_npimg(test_ds[i][1][0]), cmap='gray')
                    axs[i][j].axis('off')
                    axs[i][j].set_title('Ground truth')

                #network prediction
                elif j == 2:
                    axs[i][j].imshow(to_npimg(network_output[i][j-2][0])>0.5, cmap='gray')
                    axs[i][j].axis('off')
                    axs[i][j].set_title(network_output[i][j-2][1])
                else:
                #other output
                    axs[i][j].imshow(to_npimg(network_output[i][j-2][0]), cmap='gray')
                    axs[i][j].axis('off')
                    axs[i][j].set_title(network_output[i][j-2][1])

        plt.savefig(os.path.join(self.cp_PATH,str(self.cfgs.k_fold) + '_vis.png'))
        #plt.show()
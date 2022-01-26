'''
Create nuclei dataset.
'''

import os
import sys

import cv2 as cv
import torch
import numpy as np
import albumentations as A
from tqdm import tqdm
from dotmap import DotMap
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.utils import shuffle
from PIL import Image 

from utils import img_utils, dataAug

'''
weak aug
A.Compose(
                [A.GaussianBlur(sigma_limit=(0.0, 4.0), p=1),
                 A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
                 A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, p=1),
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
             ])   
'''

'''
heavy aug

A.Compose([ #Geometric
            A.Rotate(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent={'x': (-0.125, 0.125), 'y': (-0.125, 0.125)}, p=1),

            #Noise
            A.GaussNoise(p=0.7),
            A.GaussianBlur(sigma_limit=(0.0, 4.0), p=1),

            # #color
            A.RandomBrightnessContrast(p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1)         
])       
'''


class nucleiDataset(Dataset):
    def __init__(self, phase, dataset, mode_label, \
                stage=0, k_fold=0, random_shift=0, model_name='' \
                , cut_size=0, seed=0, data_norm='unit', color_norm=None, aug=True):
        
        self.stage = stage
        
        #image, ground truth, voronoi, pseudo label path
        rs_path = '_r' + str(random_shift)
        cut_path = 'original' if cut_size == 0 else 'cut_size_' + str(cut_size)
        norm_path = '' if color_norm == None else str(color_norm)
        
        if data_norm == 'unit':
            self.data_norm = lambda x : x/255
        elif data_norm == 'min_max':
            self.data_norm = lambda x : (x - np.min(x))/(np.max(x) - np.min(x))
        elif data_norm == 'z_mean' :
            self.data_norm = lambda x : (x - np.mean(x))/np.std(x)
        else:
            self.data_norm = lambda x : x
        
        self.img_path   = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Input_Images')
        self.gt_path    = os.path.join('.', 'datasets', dataset, cut_path, 'Labels_GT_PNG')
        self.vor_path   = os.path.join('.', 'datasets', dataset, cut_path, 'Labels_Vor'+rs_path)
        
        if stage == 0:
            self.pl_path = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Labels_' + mode_label + rs_path, 'stage0')
            self.refined_pl_path = None
            self.cf_path = None
        else:
            if stage == 1:
                self.pl_path = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Labels_' + mode_label + rs_path, 'stage0')
            else:
                self.pl_path = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Labels_' + mode_label+rs_path, 'stage'+str(stage-1), str(model_name), 'refined', 'CV'+str(k_fold))
                
            self.refined_pl_path = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Labels_' + mode_label+rs_path, 'stage'+str(stage), str(model_name), 'refined', 'CV'+str(k_fold))
            self.cf_path = os.path.join('.', 'datasets', dataset, cut_path, norm_path, 'Labels_' + mode_label+rs_path, 'stage'+str(stage), str(model_name), 'CL', 'CV'+str(k_fold))
            
        
        self.names = self._load_names(phase, dataset, k_fold, seed)
        
        if aug == True:
            self.aug = A.Compose(
                [A.GaussianBlur(sigma_limit=(0.0, 4.0), p=1),
                 A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
                 A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, p=1),
                 A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
             ])           
        else:
            self.aug = None
            
    
    def display_path(self):
        print('Image Path: ', self.img_path)
        print('Ground truth Path: ', self.gt_path)
        print('Voronoi Path: ', self.vor_path)
        
        if self.stage > 0:
            print('Pseudo label Path: ', self.pl_path)
            print('Refined label Path: ', self.refined_pl_path)
            print('Confusion matrix path: ', self.cf_path)
    
    def _display_fold(self, size_of_fold, index_list):
        for i, e in enumerate(index_list):
            if i % size_of_fold == 0 and i != 0:
                print(';', end='')
            print(f'{e:>3}', end='')
        print()

    def _load_names(self, phase, dataset, k_fold, seed):
        total_name_list = np.array([e[:-4] for e in os.listdir(self.img_path)])
        total_shuffled_index = shuffle(np.arange(len(total_name_list)), random_state=seed)
        
        size_of_fold = len(total_name_list)//10
        
        val_index = np.arange(k_fold*size_of_fold, (k_fold+1)*size_of_fold)
        test_index = np.arange((k_fold+1)%10*size_of_fold, ((k_fold+1)%10+1)*size_of_fold)
        train_index = np.delete(total_shuffled_index, np.concatenate([val_index, test_index], 0))
        
        val_index = total_shuffled_index[val_index]
        test_index = total_shuffled_index[test_index]

        if phase == 'train':
            name_list = list(total_name_list[train_index])
            print('Train:', end='')
            self._display_fold(size_of_fold, train_index)
        elif phase == 'val':
            name_list = list(total_name_list[val_index])
            print('Val:', end='')
            self._display_fold(size_of_fold, val_index)
        elif phase == 'test':
            print('Test:', end='')
            name_list = list(total_name_list[test_index])
            self._display_fold(size_of_fold, test_index)
        elif phase == 'total':
            print('Total:', end='')
            name_list = list(total_name_list)
            self._display_fold(size_of_fold, total_shuffled_index)
        
        return name_list
    
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):
        #0, ground truth
        #1, voronoi label
        #2, pseudo label, if stage>1, store stage-1 refined label
        #3, refined_label
        #4, confusion_matrix
        
        x = np.array(Image.open(os.path.join(self.img_path, self.names[idx]+'.png')))
        #remove transparency channel
        if x.shape[-1] == 4:
            x = x[:,:,:3] 
        y = []
        y.append(np.array(Image.open(os.path.join(self.gt_path, self.names[idx]+'.png')))//255)
        y.append(np.array(Image.open(os.path.join(self.vor_path, self.names[idx]+'.png')))//127)
        y.append(np.array(Image.open(os.path.join(self.pl_path, self.names[idx]+'.png')))//255)
        
        if self.stage > 0:
            y.append(np.array(Image.open(os.path.join(self.refined_pl_path, self.names[idx]+'.png')))//255)
            y.append(np.array(Image.open(os.path.join(self.cf_path, self.names[idx]+'.png')))//255)
        
        #data augmentation
        if self.aug is not None:
            augged = self.aug(image=x, masks=y)
            x = augged['image']
            y = augged['masks']
            
        #transform to pytorch tensor
        x = torch.from_numpy(x).permute(2, 0, 1).float() #HWC => CHW
        y = torch.from_numpy(np.array(y)).float()  
        assert x.shape[0] == 3    #CHW
        assert x.ndim == 3
        assert y.ndim == 3
        
        if self.stage > 0:
            assert y.shape[0] == 5
        else:
            assert y.shape[0] == 3
        
        #data normalisation
        x = self.data_norm(x)
        
        return x, y
        

def _gen_confusion_matrix(img_path, gt_path, pl_path, \
                        refined_pl_path, cf_path, network_path, \
                        data_norm, device):
    '''
    load or generate confusion matrix
    '''
    data_ids = [e[:-4] for e in os.listdir(img_path)]
    
    print(f'pre_pseudo_path: {pl_path}')
    print(f'cur_pseudo_path: {refined_pl_path}')
    print()

    os.makedirs(cf_path, exist_ok=True)
    os.makedirs(refined_pl_path, exist_ok=True)
    
    #load network
    network = torch.load(network_path, map_location=device)
    network = network[0]
    network.to(device)
    network.eval()
    
    #generate confusion matrix
    print('Generating Confusion Matrix...')
    sys.stdout.flush()
    
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):
        #load image, pseudo label from pre stage
        x_image = cv.imread(os.path.join(img_path, data_ids[i] +'.png'))
        x_image = cv.cvtColor(x_image, cv.COLOR_BGR2RGB)
        y_pre_pse = cv.imread(os.path.join(pl_path, data_ids[i] +'.png'), cv.IMREAD_UNCHANGED)//255
        
        #convert to tensor
        #HWC => CHW => BCHW => float type
        x_image = data_norm(torch.from_numpy(x_image).permute(2, 0, 1).unsqueeze(0).float()) 
        
         
        with torch.no_grad():
            out = network(x_image.to(device))
            if type(out) == tuple:
                pred = out[0].squeeze(0).detach().cpu().clone().numpy()
            else:
                pred = out.squeeze(0).detach().cpu().clone().numpy()         #CHW
            
            assert pred.ndim == 3
            assert pred.shape[0] == 1
            assert y_pre_pse.ndim == 2
            
            #generate confusion matrix based on pre pseudo labels and network output
            y_cf = img_utils.gen_confusion_matrix(pred, y_pre_pse)
            
            #generate refined new pseudo labels based on pre pseudo labels and confusion matrix
            y_refined_pl = img_utils.gen_refined_label(y_pre_pse, y_cf)
        
        #print(np.sum(y_cf - y_pre_pse))

        #save cf and refined labels
        cv.imwrite(os.path.join(refined_pl_path, data_ids[i]+'.png'), (y_refined_pl*255).astype(np.uint8))
        cv.imwrite(os.path.join(cf_path, data_ids[i]+'.png'), (y_cf*255).astype(np.uint8))

    print('Done!')

def _datasplit(X_data, y_data, k_fold, seed):
    '''
    
    split data into train_set, validation_set and test_set

    Parameters
    ----------
    @X_data:   input image
    @y_data:   ground truth 
    @k_fold: kth iteration to do k_fold cross validation
    @seed: shuffle data
    '''


    data_index = shuffle(np.arange(len(X_data)), random_state=seed)
    size_of_fold = len(data_index)//10
    train_index = []
    val_index = []
    test_index = []
    
    for i in range(10):
        if i == k_fold%10:
            val_index.extend(list(data_index[i*size_of_fold:(i+1)*size_of_fold]))
        elif i == (k_fold+1)%10:
            test_index.extend(list(data_index[i*size_of_fold:(i+1)*size_of_fold]))
        else:
            train_index.extend(list(data_index[i*size_of_fold:(i+1)*size_of_fold]))

    return X_data[train_index], y_data[train_index], X_data[val_index], y_data[val_index], X_data[test_index], y_data[test_index]


def _data_norm(data, method):
    '''
    0: X_train
    1: y_train
    2: X_val
    3: y_val
    4: X_test
    5: y_test
    '''
    my_preprocess = img_utils.get_preprocess_method(method)
    
    X_train, y_train = my_preprocess(data[0], data[1]) 
    X_val, y_val     = my_preprocess(data[2], data[3])
    X_test, y_test   = my_preprocess(data[4], data[5])
        
    return (X_train, y_train, X_val, y_val, X_test, y_test)


def _load_dataset(datains, cfgs):
    data_ids = [i[:-4] for i in os.listdir(datains.img_path)]
    if cfgs.dataset == 'MoNuSeg':
        dataH, dataW = 1000, 1000
    elif cfgs.dataset == 'TNBC':
        dataH, dataW = 512, 512
    elif cfgs.dataset == 'CPM17':
        dataH, dataW = 512, 512
        
    #read data
    X_data = np.empty((len(data_ids), dataH, dataW, 3), dtype=np.uint8)
    if cfgs.stage > 0:
        y_data = np.empty((len(data_ids), dataH, dataW, 5), dtype=np.uint8)
    else:
        y_data = np.empty((len(data_ids), dataH, dataW, 3), dtype=np.uint8)
    
    for i in range(len(data_ids)):
        img = cv.imread(os.path.join(datains.img_path, data_ids[i]+'.png'))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        X_data[i] = img
        y_data[i,:,:,0] = cv.imread(os.path.join(datains.gt_path, data_ids[i]+'.png'), \
                                                cv.IMREAD_UNCHANGED) // 255
        y_data[i,:,:,1] = cv.imread(os.path.join(datains.vor_path, data_ids[i]+'.png'), \
                                                cv.IMREAD_UNCHANGED) // 127
        y_data[i,:,:,2] = cv.imread(os.path.join(datains.pl_path, data_ids[i]+'.png'), \
                                                cv.IMREAD_UNCHANGED) // 255
        
        if cfgs.stage > 0:
            y_data[i,:,:,3] = cv.imread(os.path.join(datains.refined_pl_path, data_ids[i]+'.png'), \
                                                cv.IMREAD_UNCHANGED) // 255
            y_data[i,:,:,4] = cv.imread(os.path.join(datains.cf_path, data_ids[i]+'.png'), \
                                                cv.IMREAD_UNCHANGED) // 255
                                                  
    
    #split data into train, val, test
    X_train, y_train, X_val, y_val, X_test, y_test = \
            _datasplit(X_data, y_data, k_fold=cfgs.k_fold, seed=cfgs.seed)

    
    #training data augmentation
    X_train, y_train = dataAug.myDataAug(X_train, y_train, 4, seed=cfgs.seed)

    #data preprocessing
    data = _data_norm([X_train, y_train, X_val, y_val, X_test, y_test], cfgs.data_norm)
    
    #channel first format
    X_train, y_train, X_val, y_val, X_test, y_test = img_utils.ch_first(data)
    
    #move to torch tensor
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=cfgs.bs, shuffle=True)

    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=1)

    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=1)
    
    
    return train_dl, val_dl, test_dl, test_ds

    
def get_data(cfgs):
    
    '''
    return training data, depend on different stage
    
    Parameters:
    -----------
    @stage: int type, should be natural number
            '0', initial training
            '1', apply CL based on the result of '0'st training
            '2', apply CL based on the result of '1'nd training
            so on...
            
            ###
            When we use number x that is larger than 0,
            make sure we already train all networks with stage < x, and use the same
            k_fold value. That is, if we want to train with stage = 2, and k_fold=3,
            We should train with stage=1,0 and k_fold=3, 
            Also, keep the model name consistent with pretrained network.
            
            The CL requires pseudo ground truth and pretrained network to generate prediction
            for label correction
            ##### 
    @cp_time: debug value, str type
            'final': load network of final epochs
            'best': load network performance based on validation dataset
    @dataset: which data to load
    @k_fold: k'th fold cross validation training
    @random_shift: int type, shift point annotation to add noise
                   randomly move each point in horizontal and vertical direction. 
                   range from [-random_shift, +random_shift]
    @model_name: str type, model name of pretrained network
    @mode_label:'Sp': use superpixel to train Segmentation network
                'Kmeans': use kmeans cluster to...
                'AbKmeans_05': Anchor based kmeans cluster with rectangle ratio 1:2 
                'AbKmeans_1':  Anchor based kmeans cluster with rectangle ratio 1:1
                'AbKmeans_2':  Anchor based kmeans cluster with rectangle ratio 2:1
                'AbKmeans_comb':  Anchor based kmeans cluster with combination of previous three
    @cut_size: integer N, cut data into ceiling(data_H/N) part, each part with a size of N*N, 
    @bs: batch size
    @seed: seed for dataset split, data augmentation 
           default: 12 
    @data_norm: apply data normalisation on input images
                process on along each image rather than whole dataset
                'unit': x = x/255,
                'min_max': x = (x-min(x))/(max(x) - min(x)),
                'z_mean': x = (x-mean(x))/std(x)
                None: doesn't apply any data_norm method
                default: 'unit'
    @device: running device for generating the confusion matrix
            'cpu':  ...
            'cuda': ...
    @color_norm: color normalisation on input image, generate new data
                   'norm1':...
                   'norm2':...
    '''
    legacy = False
    
    dscfgs = {
        'stage'         : cfgs.stage,
        'dataset'       : cfgs.dataset, 
        'k_fold'        : cfgs.k_fold,
        'random_shift'  : cfgs.random_shift,
        'model_name'    : cfgs.model_name,
        'mode_label'    : cfgs.mode_label,
        'cut_size'      : cfgs.cut_size,
        'seed'          : cfgs.seed,
        'data_norm'     : cfgs.data_norm,
        'color_norm'    : cfgs.color_norm,
    }
    
    total_ds = nucleiDataset(phase='total', **dscfgs)   #for debug

    train_ds = nucleiDataset(phase='train', **dscfgs)
    val_ds = nucleiDataset(phase='val', **dscfgs, aug=False)
    test_ds = nucleiDataset(phase='test', **dscfgs, aug=False)
    print()

    total_ds.display_path()
    print()

    if cfgs.stage > 0:
        _gen_confusion_matrix(img_path=total_ds.img_path, gt_path=total_ds.gt_path, \
                                pl_path=total_ds.pl_path, refined_pl_path=total_ds.refined_pl_path, \
                                cf_path=total_ds.cf_path, network_path=cfgs.network_path, \
                                data_norm=total_ds.data_norm, device=cfgs.device)
    
    assert legacy is False
    
    if legacy is True:
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=cfgs.bs)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)
    
    else:
        train_dl, val_dl, test_dl, test_ds = _load_dataset(total_ds, cfgs)

    return train_dl, val_dl, test_dl, test_ds

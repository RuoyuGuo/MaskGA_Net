'''
generate superpixel annotations, point annotations. etc..
use this before you train network
'''

import os
import sys
from math import ceil
from copy import deepcopy

import cv2 as cv
import numpy as np
import click
from tqdm import tqdm
from dotmap import DotMap
from skimage import morphology

from utils import img_utils
from img_norm import norm

#create path for convenience
class pathConstructor():
    def __init__(self, dataset, mode_label, random_shift, color_norm, cut_size):
        '''
        self.xx: final data path, after norm, after divided
        self.xx_origin:  data path with original size
        
        self.img                    :  influence by color norm
        self.gt_origin              :  influenced by nothing
        self.voronoi_origin         :  influenced by random shift
        self.pseudo_label_origin    :  influenced by random shift, color norm
        '''
        rs_path = '_r' + str(random_shift)
        cut_path = 'original' if cut_size == 0 else 'cut_size_' + str(cut_size)
        norm_path = '' if color_norm == None else str(color_norm)
        
        self.dataset = dataset
        self.cut_size = cut_size
        
        self.img = os.path.join('.', 'datasets', dataset, cut_path , norm_path, 'Input_Images')        
        self.img_origin = os.path.join('.', 'datasets', dataset, 'original', 'Input_Images')
        self.img_norm = os.path.join('.', 'datasets', dataset, 'original' , norm_path, 'Input_Images')        
        
        self.gt = os.path.join('.', 'datasets', dataset, cut_path, 'Labels_GT_PNG')
        self.gt_origin =  os.path.join('.', 'datasets', dataset, 'original', 'Labels_GT_PNG')

        self.voronoi = os.path.join('.', 'datasets', dataset, cut_path, 'Labels_Vor'+rs_path)
        self.voronoi_origin = os.path.join('.', 'datasets', dataset, 'original', 'Labels_Vor'+rs_path)
        
        self.pseudo_label = os.path.join('.', 'datasets', dataset, cut_path, norm_path, \
                                    'Labels_' + mode_label + rs_path, 'stage0')
        self.pseudo_label_origin = os.path.join('.', 'datasets', dataset, 'original', norm_path, \
                                    'Labels_' + mode_label + rs_path, 'stage0')

        self.xml = None
        if self.dataset == 'MoNuSeg':
            self.xml = os.path.join('.', 'datasets', dataset, 'original', 'Labels_GT_XML')        
       
       
        if cut_size == 0 and color_norm is None:
            assert self.img == self.img_origin
            assert self.img == self.img_norm
            assert self.gt == self.gt_origin
            assert self.voronoi == self.voronoi_origin
            assert self.pseudo_label == self.pseudo_label_origin
            
        
        self._create_folder()
        
    #display, for debug
    def show(self):
        print('Image: ', self.img)
        print('Image_origin: ', self.img_origin)
        print('Image_norm: ', self.img_norm)
        print('GT: ', self.gt)
        print('GT_origin: ', self.gt_origin)
        print('Vor: ', self.voronoi)
        print('Vor_origin: ', self.voronoi_origin)
        print('Pse: ', self.pseudo_label)
        print('Pse_origin: ', self.pseudo_label_origin)
        print('Xml: ', self.xml)
        
    def _create_folder(self):
        os.makedirs(self.voronoi_origin, exist_ok=True)
        os.makedirs(self.pseudo_label_origin, exist_ok=True)
        
        #create new folder if cut images
        if self.cut_size > 0:
            os.makedirs(self.img, exist_ok=True)
            os.makedirs(self.gt, exist_ok=True)
            os.makedirs(self.voronoi, exist_ok=True)
            os.makedirs(self.pseudo_label, exist_ok=True)
        
        if self.dataset == 'MoNuSeg':
            os.makedirs(self.gt_origin, exist_ok=True)
    
def _divided(img, H, N):
    '''
    H: image height and width
    N: cut_size
    '''
    num = ceil(H/N)
    stride = (H-N)//(num-1)
    img_patchs = []
    for i in range(num):
        for j in range(num):
            img_patchs.append(img[stride*i:stride*i+N, stride*j:stride*j+N]) 

    return img_patchs


def dataset_MoNuSeg(rng,cfgs):
    '''
    Generate supervision for MoNuSeg dataset, 
    for details, see my_dataset.py
    '''
    
    #define path
    HW = 1000
    pathkit = pathConstructor(dataset='MoNuSeg', \
                                random_shift=cfgs.random_shift, \
                                mode_label=cfgs.mode_label, \
                                color_norm=cfgs.color_norm, \
                                cut_size=cfgs.cut_size)

    data_ids = [i[:-4] for i in os.listdir(pathkit.img_origin)]
    assert len(data_ids) == 30
    X_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)
    y_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)   

    if cfgs.color_norm != None:
        print(f'Normalising images...')        
        norm.norm(source_image_dir=pathkit.img_origin, result_dir=pathkit.img_norm, method=str(cfgs.color_norm))

    print(f'Generating {cfgs.mode_label} supervision...')
    sys.stdout.flush()
    
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):
        #read image
        if cfgs.color_norm != None:
            img = cv.imread(os.path.join(pathkit.img_norm, data_ids[i]+'.png'))
        else:
            img = cv.imread(os.path.join(pathkit.img_origin, data_ids[i]+'.png'))            
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        #generate
        X_data[i] = img
        y_data[i] = img_utils.gen_pseudo_labels_MoNuSeg(X_data[i], \
                    os.path.join(pathkit.xml, data_ids[i]+'.xml'), \
                    cfgs.mode_label, cfgs.random_shift, rng)
                    
        if cfgs.mode_label == 'Kmeans' and cfgs.kernel != 'none':
            print('Applying closing operation...')
            y_data[i,:,:,2] = morphology.closing(y_data[i,:,:,2])
    
        #write origin images, labels
        cv.imwrite(os.path.join(pathkit.gt_origin, data_ids[i]+'.png'), y_data[i,:,:,0]*255)              #gt
        cv.imwrite(os.path.join(pathkit.voronoi_origin, data_ids[i]+'.png'), y_data[i,:,:,1]*127)     #voronoi
        cv.imwrite(os.path.join(pathkit.pseudo_label_origin, data_ids[i]+'.png'), y_data[i,:,:,2]*255)    #pseudo
        
        if cfgs.cut_size != 0:
            #write input images
            for i_patch, patch in enumerate(_divided(X_data[i], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.img, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)
            
            #write ground truth
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,0], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.gt, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)

            #write voronoi 
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,1]*127, HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.voronoi, data_ids[i]+'_'+str(i_patch)+'.png'),\
                                    patch)
                                    
            #write pseudo labels
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,2], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.pseudo_label, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                        patch)

    print('Done!')


def dataset_TNBC(rng, cfgs):
    '''
    Generate supervision for TNBC dataset, 
    for details, see my_dataset.py
    '''
    
    #define path
    HW = 512
    pathkit = pathConstructor(dataset='TNBC', \
                                random_shift=cfgs.random_shift, \
                                mode_label=cfgs.mode_label, \
                                color_norm=cfgs.color_norm, \
                                cut_size=cfgs.cut_size)

    data_ids = [i[:-4] for i in os.listdir(pathkit.img_origin)]
    assert len(data_ids) == 50
    X_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)
    y_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)   

    if cfgs.color_norm != None:
        print(f'Normalising images...')        
        norm.norm(source_image_dir=pathkit.img_origin, result_dir=pathkit.img_norm, method=str(cfgs.color_norm))

    print(f'Generating {cfgs.mode_label} supervision...')
    sys.stdout.flush()
   
    
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):

        #read image and ground truth
        ground_truth = cv.imread(os.path.join(pathkit.gt_origin, data_ids[i]+'.png'), cv.IMREAD_GRAYSCALE)    
        ground_truth = np.clip(ground_truth, 0, 1)
        
        if cfgs.color_norm != None:
            img = cv.imread(os.path.join(pathkit.img_norm, data_ids[i]+'.png'))
        else:
            img = cv.imread(os.path.join(pathkit.img_origin, data_ids[i]+'.png'))            
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
        #generate
        X_data[i] = img
        y_data[i] = img_utils.gen_pseudo_labels_TNBC(X_data[i], ground_truth, \
                                                    cfgs.mode_label, cfgs.random_shift, rng)
                                                    
        if cfgs.mode_label == 'Kmeans' and cfgs.kernel != 'none':
            print('Applying closing operation...')
            y_data[i,:,:,2] = morphology.closing(y_data[i,:,:,2])
        
        #write origin images, labels
        cv.imwrite(os.path.join(pathkit.voronoi_origin, data_ids[i]+'.png'), y_data[i,:,:,1]*127)     #voronoi
        cv.imwrite(os.path.join(pathkit.pseudo_label_origin, data_ids[i]+'.png'), y_data[i,:,:,2]*255)    #pseudo
        
        if cfgs.cut_size != 0:
            #write input images
            for i_patch, patch in enumerate(_divided(X_data[i], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.img, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)
            
            #write ground truth
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,0], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.gt, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)

            #write voronoi 
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,1]*127, HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.voronoi, data_ids[i]+'_'+str(i_patch)+'.png'),\
                                    patch)
                                    
            #write pseudo labels
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,2], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.pseudo_label, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                        patch)

    print('Done!')


def _dataset_CPM17(rng, cfgs):
    '''
    Generate supervision for CPM17 dataset, 
    for details, see my_dataset.py
    '''
    
    #define path
    HW = 512
    pathkit = pathConstructor(dataset='CPM17', \
                                random_shift=cfgs.random_shift, \
                                mode_label=cfgs.mode_label, \
                                color_norm=cfgs.color_norm, \
                                cut_size=cfgs.cut_size)

    data_ids = [i[:-4] for i in os.listdir(pathkit.img_origin)]
    assert len(data_ids) == 50
    X_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)
    y_data = np.empty((len(data_ids), HW, HW, 3), dtype=np.uint8)   

    if cfgs.color_norm != None:
        print(f'Normalising images...')        
        norm.norm(source_image_dir=pathkit.img_origin, result_dir=pathkit.img_norm, method=str(cfgs.color_norm))

    print(f'Generating {cfgs.mode_label} supervision...')
    sys.stdout.flush()
   
    
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):

        #read image and ground truth
        ground_truth = cv.imread(os.path.join(pathkit.gt_origin, data_ids[i]+'.png'), cv.IMREAD_GRAYSCALE)    
        ground_truth = np.clip(ground_truth, 0, 1)
        
        if cfgs.color_norm != None:
            img = cv.imread(os.path.join(pathkit.img_norm, data_ids[i]+'.png'))
        else:
            img = cv.imread(os.path.join(pathkit.img_origin, data_ids[i]+'.png'))            
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
        #generate
        X_data[i] = img
        y_data[i] = img_utils.gen_pseudo_labels_CPM17(X_data[i], ground_truth, \
                                                    cfgs.mode_label, cfgs.random_shift, rng)
                                                    
        if cfgs.mode_label == 'Kmeans' and cfgs.kernel != 'none':
            print('Applying closing operation...')
            y_data[i,:,:,2] = morphology.closing(y_data[i,:,:,2])
            
        #write origin images, labels
        cv.imwrite(os.path.join(pathkit.voronoi_origin, data_ids[i]+'.png'), y_data[i,:,:,1]*127)     #voronoi
        cv.imwrite(os.path.join(pathkit.pseudo_label_origin, data_ids[i]+'.png'), y_data[i,:,:,2]*255)    #pseudo
        
        if cfgs.cut_size != 0:
            #write input images
            for i_patch, patch in enumerate(_divided(X_data[i], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.img, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)
            
            #write ground truth
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,0], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.gt, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                    patch)

            #write voronoi 
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,1]*127, HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.voronoi, data_ids[i]+'_'+str(i_patch)+'.png'),\
                                    patch)
                                    
            #write pseudo labels
            for i_patch, patch in enumerate(_divided(y_data[i,:,:,2], HW, cfgs.cut_size)):
                cv.imwrite(os.path.join(pathkit.pseudo_label, data_ids[i]+'_'+str(i_patch)+'.png'), \
                                        patch)

    print('Done!')
    
@click.command()
@click.option('--dataset', required=True, \
                help='Prepare supervision on which dataset')
@click.option('--random_shift', type=click.IntRange(0), default=0, \
                help='Move range(in pixel), randomly shift each point annotation')
@click.option('--seed', type=click.IntRange(0), default=12, \
                help='Seed for random shift')
@click.option('--color_norm', default=None, type=click.Choice(['norm1']), \
                help='Color normalisation method on input images')
@click.option('--mode_label', type=click.Choice(['Sp', 'Kmeans', 'AbKmeans_05', 'AbKmeans_1', 'AbKmeans_2', 'AbKmeans_comb']), \
                default='Sp', \
                help='Type of pseudo label to generate')
@click.option('--cut_size', type=click.IntRange(0), default=0, \
                help='Cut image into ceil(ImageResolution/cut_size)**2 pieces to save memory, set to 0 to avoid cut')
@click.option('--kernel', type=click.Choice(['defualt', 'none']), default='none', \
                help='kernel for closing operation when [--mode_label Kmeans]')
def main(**kwargs):
    """'Generate voronoi labels, pseudo labels for training.
    '
    
    #########################################################
    Examples (Generate Superpixel label on MoNuSeg, TNBC, CPM17 dataset and dont move point annotation):

    \b
    python supervision_prepare.py --dataset MoNuSeg
    
    \b
    python supervision_prepare.py --dataset TNBC
    
    \b
    python supervision_prepare.py --dataset CPM17
    
    #########################################################
    Examples (Generate Superpixel label on MoNuSeg and move point annotation within 3 pixels):

    \b
    python supervision_prepare.py --dataset MoNuSeg --random_shift 3
    
    
    #########################################################
    Examples (Generate anchor based kmeans label on TNBC and dont move point annotation):

    \b
    python supervision_prepare.py --dataset TNBC --mode_label AbKmeans_comb    
    
    
    ########################################################
    Examples (Generate Sp labels on MoNuSeg with cut_size=512, divided image into 4 parts, 
    this will also generate original size of labels):

    \b
    python supervision_prepare.py --dataset MoNuSeg --cut_size 512 
    
    
    ########################################################
    Examples (Generate Sp labels on MoNuSeg with normalisation, may affect pseudo labels):

    \b
    python supervision_prepare.py --dataset MoNuSeg --color_norm norm1
    
    """
    rng = np.random.default_rng(seed=kwargs['seed'])
    cfgs = DotMap(kwargs, _dynamic=False)
    
    #cut_size check
    if cfgs.cut_size%2 != 0:
        raise UserError("[cut_size] should be the power of 2")
    if cfgs.dataset == 'MoNuSeg' and cfgs.cut_size >= 1000:
        raise UserError("[cut_size] should be less than image size (1000)")
    if cfgs.dataset == 'TNBC' and cfgs.cut_size >= 512:
        raise UserError("[cut_size] should be less than image size (512)")
    if cfgs.dataset == 'CPM17' and cfgs.cut_size >= 512:
        raise UserError("[cut_size] should be less than image size (512)")
    
    if cfgs.dataset == 'MoNuSeg':
        dataset_MoNuSeg(rng=rng, cfgs=cfgs)
    elif cfgs.dataset == 'TNBC': 
        dataset_TNBC(rng=rng, cfgs=cfgs)
    elif cfgs.dataset == 'CPM17':
        dataset_CPM17(rng=rng, cfgs=cfgs)
    else:
        print('Sorry current code doesnt support custom dataset')

if __name__ == '__main__':
    main()
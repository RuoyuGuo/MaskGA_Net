'''
Data Augmentation
'''

import cv2 as cv
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import sys

from tqdm import tqdm

#print(ia.__version__)

def myDataAug(X_data, Y_data, its, seed=12):
    size = len(X_data)
    _, W, H, C_x = X_data.shape
    _, _, _, C_y = Y_data.shape
    ia.seed(seed)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 4.0)),    #Random GaussianBlur from sigma 0 to 4
        iaa.AddToHueAndSaturation((-10, 10), per_channel=True),      #color jittering
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),   #affine transalte (affects segmaps)
        iaa.Fliplr(0.5),              #50 % to flip horizontally (affects segmaps)
        iaa.Flipud(0.5),              #50 % to flip vertically  (affects segmaps)
        ], random_order=True)
    

    '''
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 4.0)),    #Random GaussianBlur from sigma 0 to 4
        iaa.AddToHueAndSaturation((-10, 10), per_channel=True),      #color jittering
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),   #affine transalte (affects segmaps)
        iaa.Fliplr(0.5),              #50 % to flip horizontally (affects segmaps)
        iaa.Flipud(0.5),              #50 % to flip vertically  (affects segmaps)
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True)
        ], random_order=True)
    '''   
    #heavy
    '''
    seq = iaa.Sequential([
        sometimes(iaa.GaussianBlur(sigma=(0.0, 4.0))),    #Random GaussianBlur from sigma 0 to 4
        sometimes(iaa.AddToHueAndSaturation((-10, 10), per_channel=True)),      #color jittering
        sometimes(iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),   #affine transalte (affects segmaps)
        iaa.Fliplr(0.5),              #50 % to flip horizontally (affects segmaps)
        iaa.Flipud(0.5),              #50 % to flip vertically  (affects segmaps)
        #iaa.Rotate((-45, 45))  # rotate by -45 to 45 degrees (affects segmaps)
        sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True)),
        #sometimes(iaa.ChannelShuffle(1)),
        #sometimes(iaa.SigmoidContrast(gain=(5, 8), cutoff=(0.4, 0.7)))
        ], random_order=True)
    '''
    '''
    seq = iaa.Sometimes(0.5, iaa.Sequential([
        #contrast
        iaa.CLAHE(tile_grid_size_px=(3, 13)),               #light
        #iaa.AllChannelsCLAHE(tile_grid_size_px=(3, 13) )     #heavy
        #iaa.SigmoidContrast(gain=(5, 8), cutoff=(0.4, 0.7)),

        #Convolutional operation
        iaa.Sharpen(alpha=(0, 1)),
        #iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),

        #noise
        #iaa.AdditiveGaussianNoise(scale=(0, 0.2*255), per_channel=True),
        iaa.SaltAndPepper(0.1),

        #blur
        iaa.GaussianBlur(sigma=(0.0, 4.0)),
        #iaa.AverageBlur(k=(3, 13)),
        #iaa.MotionBlur(k=(3, 13)),

        #Color
        #iaa.AddToBrightness((-30, 30)),
        iaa.AddToHueAndSaturation((-20, 20), per_channel=True),
        iaa.ChannelShuffle(1),

        #Geometric
        iaa.Fliplr(1),              
        iaa.Flipud(1),           
        iaa.Rot90((0, 3)),
        iaa.Affine(#scale=(0.7, 1),                                                    #heavy
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},           #heavy 
                #shear=(-20, 20),                                                      #heavy
                mode='constant'),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),                                 #heavy
        iaa.ElasticTransformation(alpha=(0, 30), sigma=(4, 6))],                      #heavy
        random_order=True))
    '''

    X_data_augs = np.zeros((its*size, W, H, C_x), dtype=np.uint8)
    Y_data_augs = np.zeros((its*size, W, H, C_y), dtype=np.uint8)

    #print('Augmenting data...')
    #sys.stdout.flush()
    for i in tqdm(range(its), total=its):
        X_data_aug, Y_data_aug = seq(images=X_data, segmentation_maps=Y_data)
        X_data_augs[i*size: (i+1)*size] = X_data_aug
        Y_data_augs[i*size: (i+1)*size] = Y_data_aug
    #print('Done!')

    return X_data_augs, Y_data_augs

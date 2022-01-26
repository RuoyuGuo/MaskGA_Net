"""
An example for histological images color normalization based on the adaptive color deconvolution as described in the paper:
https://github.com/Zhengyushan/adaptive_color_deconvolution

Yushan Zheng, Zhiguo Jiang, Haopeng Zhang, Fengying Xie, Jun Shi, and Chenghai Xue.
Adaptive Color Deconvolution for Histological WSI Normalization.
Computer Methods and Programs in Biomedicine, v170 (2019) pp.107-120.

"""
import os
import cv2
import numpy as np

from img_norm.norm1.stain_normalizer import StainNormalizer

def norm(source_image_dir, template_dir, result_dir):
    # disable GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # load template images
    template_list = os.listdir(template_dir)
    temp_images = np.asarray([cv2.imread(os.path.join(template_dir, name)) for name in template_list])
    
    # extract the stain parameters of the template slide
    normalizer = StainNormalizer()
    normalizer.fit(temp_images[:,:,:,[2,1,0]]) #BGR2RGB
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # normalization
    slide_list = os.listdir(source_image_dir)
    print('Normalising...')
    for s in slide_list:
        #image_list = os.listdir(source_image_dir)
        images = np.asarray([cv2.imread(os.path.join(source_image_dir, s))])
    
        ## color transform
        results = normalizer.transform(images[:,:,:,[2,1,0]]) #BGR2RGB
        # display
        for i, result in enumerate(results):
            # cv2.imwrite(os.path.join(result_dir, s[:-4] + '_origin.png'.format(i)), images[i])    #original image
            cv2.imwrite(os.path.join(result_dir, s[:-4] + '.png'.format(i)) , result[:,:,[2,1,0]]) #RGB2BGR
    
        ## h&e decomposition
        # he_channels = normalizer.he_decomposition(images[:,:,:,[2,1,0]], od_output=True) #BGR2RGB
        # # debug display
        # for i, result in enumerate(he_channels):
        #     cv2.imwrite(os.path.join(result_dir, s[:-4] + '_h.png'.format(i)), result[:,:,0]*128)
        #     cv2.imwrite(os.path.join(result_dir, s[:-4] + '_e.png'.format(i)), result[:,:,1]*128)
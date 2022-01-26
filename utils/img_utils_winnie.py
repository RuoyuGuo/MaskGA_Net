import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
import warnings
import torch
import torch.nn as nn
import scipy.stats as st

from torch.nn import functional as F
from utils import settings
from cleanlab import pruning 
from skimage import segmentation
from sklearn import cluster
from skimage import morphology
from skimage.draw import rectangle_perimeter, rectangle

def _get_centroid_1(nucleus_contour):
    '''
    return centroid coordinate of a nucleus,
    manually, average of all coordinates
    
    Parameters
    ----------
    @nucleus_contour: a numpy array of a series of coordinates
    '''
    
    length = len(nucleus_contour)
    sum_x = np.sum(nucleus_contour[:, 0])
    sum_y = np.sum(nucleus_contour[:, 1])
    
    return int(sum_x/length), int(sum_y/length)

def _get_centroid_2(cnt):
    '''
    return centroid coordinate of a nucleus
    use cv moment
    
    Parameters
    ----------
    @cnt: a N * 1 * 2 numpy array, where N is the number of pixel in contour
    '''
    
    M = cv.moments(cnt)
    
    cX = 0
    cY = 0
    
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
    return cX, cY

def _my_superpixel_segments(img, vor_annos, data_name, compactness=30):
    '''
    generate superpixel labels of the given image, only superpixels that contains
    point annotations are  kept.
    '''
    H, W, _ = img.shape
    temp_vor_annos = vor_annos == 1
    
    if data_name == 'MoNuSeg':
        if np.sum(temp_vor_annos) < 1000:
            n_segments = 2000
        elif np.sum(temp_vor_annos) < 1500:
            n_segments = 3000
        else:
            n_segments = 5000
    elif data_name == 'CoNSeP':
        if np.sum(temp_vor_annos) < 1000:
            n_segments = 2000
        elif np.sum(temp_vor_annos) < 1500:
            n_segments = 3000
        else:
            n_segments = 5000
    elif data_name == 'TNBC':
        if np.sum(temp_vor_annos) < 200:
            n_segments = 700
        else:
            n_segments = 1000
    elif data_name == 'CMP17':
        if np.sum(temp_vor_annos) < 200:
            n_segments = 500
        else:
            n_segments = 1000

    #get superpixel segmentation
    segments = segmentation.slic(img, n_segments=n_segments, compactness=30, start_label=1)
    if np.min(segments) == 0:
        segments += 1

    mask = np.zeros((H, W), dtype=np.uint8)

    #only keep superpixel contains points
    for i in range(np.min(segments), np.max(segments)+1):
        if np.sum((segments==i)*temp_vor_annos) > 0:
            mask += (segments==i)
    
    return np.clip(mask.astype(np.uint8),0,1)

def _my_kmeans_segments(img, vor_annos):
    '''
    generate kmean cluster segmentation (pseudo)
    '''
    
    H, W, _ = img.shape
    num_of_nuclei = np.sum(vor_annos==1)
    num_of_bg = np.sum(vor_annos==2)

    #three classes
    center_nuclei = np.sum(np.expand_dims(vor_annos==1, -1) * img, axis=(0, 1))/num_of_nuclei
    center_bg     = np.sum(np.expand_dims(vor_annos==2, -1) * img, axis=(0, 1))/num_of_bg
    center_unknown = (center_nuclei+center_bg)/2
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmean = cluster.KMeans(3,np.array([center_nuclei, center_bg, center_unknown])).fit(img.reshape(-1, 3))
    segment = kmean.labels_.reshape(H, W) == 0
    
    return segment
    
    
def mix_match(img, bound, center, ratio, rr2, cc2, alpha):
    center_nuclei = img[center]
    dist = bound // 3
    aug_img = np.copy(img)
    scale = 1

    while scale <= bound:
        if scale <= dist:
            weight = alpha
        elif scale <= dist*2:   
            weight = 2*alpha - 1
        else:
            weight = 1-alpha

        if ratio == 0.5:
            start = center[0]-int(1.5*scale), center[1]-scale
            end = center[0]+int(1.5*scale), center[1]+scale
        elif ratio == 1:
            start = center[0]-scale, center[1]-scale
            end = center[0]+scale, center[1]+scale     
        elif ratio == 2:
            start = center[0]-scale, center[1]-int(1.5*scale)
            end = center[0]+scale, center[1]+int(1.5*scale)    

        rr, cc = rectangle_perimeter(start,end,shape=aug_img.shape)
        aug_img[rr, cc] = center_nuclei*weight + aug_img[rr, cc]*(1-weight)
                
        scale += 1

    H,W,C = aug_img[rr2, cc2].shape   
    center_bg = np.mean(aug_img[rr, cc], axis=0)
    kmean = cluster.KMeans(2,np.array([center_nuclei, center_bg])).fit(aug_img[rr2, cc2].reshape(-1, 3)) 
    segment = kmean.labels_.reshape(H, W) == 0
        
    return segment


def _AbKmeans(img, center, ratio, bound, update_num, threshold, patience, alpha, verbose=False):
    scale = 7   
    final_scale = None
    fg_bg_ratio = 0
    count_segment = 0
    final_rr, final_cc, final_rr2, final_cc2 = None, None, None, None
    img_seg = np.zeros((img.shape[0], img.shape[1]))

    while scale < bound :
        if ratio == 0.5:
            start = center[0]-int(1.5*scale), center[1]-scale
            end = center[0]+int(1.5*scale), center[1]+scale
        elif ratio == 1:
            start = center[0]-scale, center[1]-scale
            end = center[0]+scale, center[1]+scale     
        elif ratio == 2:
            start = center[0]-scale, center[1]-int(1.5*scale)
            end = center[0]+scale, center[1]+int(1.5*scale)    

        if verbose is True:
            print(f'start: {start}, end: {end}')

        rr, cc = rectangle_perimeter(start,end,shape=img.shape)
        rr2, cc2 = rectangle(start,end,shape=img.shape)
        region = img[rr2, cc2]
        H,W,C = region.shape

        center_nuclei = img[center]
        center_bg = np.mean(img[rr, cc], axis=0)

        kmean = cluster.KMeans(2,np.array([center_nuclei, center_bg])).fit(region.reshape(-1, 3))
        segment = kmean.labels_.reshape(H, W) == 0

        fg_nums = np.sum(segment)
        bg_nums = H*W-fg_nums
        if verbose is True:
            print(f'new percent: {fg_nums/bg_nums:.4f}')

        if fg_nums/bg_nums > threshold or fg_bg_ratio == 0:
            final_scale = scale
            final_rr,  final_cc  = rr, cc
            final_rr2, final_cc2 = rr2, cc2
            img_seg[rr2, cc2] = segment
            img_seg = morphology.binary_opening(img_seg)
            img_seg = morphology.binary_closing(img_seg)
            scale += update_num
            if verbose is True:
                print(f'scale increase to {scale}')
            fg_bg_ratio = fg_nums/bg_nums
        elif patience > 0:
            patience -= 1
            scale += update_num
            if verbose is True:
                print(f'scale increase to {scale}')
        else:
            break
    
    img_seg[final_rr2, final_cc2] = mix_match(img, final_scale, center, ratio, final_rr2, final_cc2, alpha)

    return img_seg, final_rr2, final_cc2, img_seg[final_rr2, final_cc2].shape
    

def _my_AbKmeans_segments(img, vor_annos, ratio, local=False):
    '''
    generate Anchor based Kmean cluster segmentation 
    '''
    
    options = [{'ratio': 0.5, 'bound': 13, 'threshold':0.7, 'patience':1,
            'update_num':2, 'alpha':0.75, 'verbose':False},
             {'ratio': 1, 'bound': 17, 'threshold':0.7, 'patience':1,
            'update_num':2, 'alpha':0.75, 'verbose':False},
           {'ratio': 2, 'bound': 13, 'threshold':0.7, 'patience':1,
            'update_num':2, 'alpha':0.75, 'verbose':False}]
            
    if ratio == '05':
        ops = options[0]
    elif ratio == '1':
        ops = options[1]
    elif ratio == '2':
        ops = options[2]
        
    H, W, _ = img.shape
    img_seg = np.zeros((H, W))
    bbox_seg = np.zeros((H, W))
    bbox = []
    
    #find all points
    coors = np.where(vor_annos==1)
    coors = tuple(zip(*coors))
    
    #for each point, apply AbKmeans on it. and compute the union of them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for coor in coors:
            if ratio == 'comb':    
                img_seg_next_05, rr_05, cc_05, shape_05 = _AbKmeans(img, coor, **options[0])
                img_seg_next_1, rr_1, cc_1, shape_1 = _AbKmeans(img, coor, **options[1])
                img_seg_next_2, rr_2, cc_2, shape_2 = _AbKmeans(img, coor, **options[2])
                        
                if np.argmax([shape_05[0]*shape_05[1], shape_1[0]*shape_1[1], shape_2[0]*shape_2[1]]) == 0:   
                    img_seg = np.logical_or(img_seg, img_seg_next_05)
                    bbox.append([rr_05, cc_05])
        
                elif np.argmax([shape_05[0]*shape_05[1], shape_1[0]*shape_1[1], shape_2[0]*shape_2[1]]) == 1:   
                    img_seg = np.logical_or(img_seg, img_seg_next_1)
                    bbox.append([rr_1, cc_1])
                else:
                    img_seg = np.logical_or(img_seg, img_seg_next_2)
                    bbox.append([rr_2, cc_2])
    
            else:
                img_seg_next, rr, cc, shape = _AbKmeans(img, coor, **ops)
                img_seg = np.logical_or(img_seg, img_seg_next)
                bbox.append([rr, cc])
    
    for e in bbox:
        bbox_seg[e[0], e[1]] = 1

    if local is True:
        return bbox
    
    return np.moveaxis(np.array([img_seg, bbox_seg]), 0, 2)

def _add_random_shift(ctd, r, border):
    '''
    randomly move point in horizontal and vertical direction.
    range   [ctd[0]-r, ctd[0]+r]
            [ctd[1]-r, ctd[1]+r]
    '''
    x1, x2 = ctd[0], ctd[1]
    
    
    while True:
        t1 = x1 + np.random.randint(-r, r+1)
        #print(t1)
        if t1 >= 0 and t1 < border and t1 != x1:
            break

    while True:
        t2 = x2 + np.random.randint(-r, r+1)
        #print(t2)
        if t2 >= 0 and t2 < border and t2 != x2:
            break        

    return t1, t2
    


def gen_pseudo_labels_MoNuSeg(img, path, mode_label, random_shift):
    '''
    generate label data for training
    
    Parameters
    ----------
    @img: input image
    @path: label path of the xml file
    
    Parameters
    ----------
    @A 4 channel image
    @channel 0, store ground truth
    @channel 1, store centroid and voronoi boundary
    @channel 2, store selected
    '''

    #read xml file
    tree = ET.parse(path)
    root = tree.getroot()
    annos = []
    
    #get edge annotation coordinates
    for region in root.iter('Region'):
        annos.append(region)

    cnts = []                                         #store contour coordinates of each nucleus

    label_channel_0 = np.zeros((1000, 1000), dtype=np.uint8)     #channel 0, store ground truth
    label_channel_1 = np.zeros((1000, 1000), dtype=np.uint8)     #channel 1, store centroid and voronoi boundary
    label_channel_2 = np.zeros((1000, 1000, 2), dtype=np.uint8)  if mode_label in settings.AbKmeans_family \
                        else  np.zeros((1000, 1000), dtype=np.uint8)
                                                               #channel 2, store selected pseudo labels
                                                               
    subdiv = cv.Subdiv2D((0, 0, 1000, 1000))                   #store voronoi diagram
        
    #get contour and centroid of each nucleus
    for region in annos:
        cnt = []

        for vertex in region.iter('Vertex'):
            cnt.append((np.float32(vertex.attrib['X']), np.float32(vertex.attrib['Y'])))

        cnt = np.array(cnt)
        cnt = cnt[:, np.newaxis, :]
        ctd = _get_centroid_2(cnt)
        cnts.append(cnt.astype(np.int32))
        
        if ctd[0] >= 0 and ctd[0] < 1000 and ctd[1] >= 0 and ctd[1] < 1000:
            if random_shift > 0:
                ctd = _add_random_shift(ctd, random_shift, 1000)
            label_channel_1[ctd[1], ctd[0]] = 1
            subdiv.insert(ctd)

    #draw voronoi boundary  channel_1
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for e in facets:
        cv.polylines(label_channel_1, [e.astype(np.int)], True, 2)
    
    
    #draw ground truth  channel_0
    cv.drawContours(label_channel_0, cnts, -1, 1, -1)


    #draw pseudo ground truth  channel_2
    pseudo_gt_selector = {
        'Sp': lambda x, y: _my_superpixel_segments(x, y, 'MoNuSeg'), 
        'Kmeans': lambda x, y: _my_kmeans_segments(x, y), 
        'AbKmeans_05': lambda x, y: _my_AbKmeans_segments(x, y, '05'), 
        'AbKmeans_1': lambda x, y: _my_AbKmeans_segments(x, y, '1'), 
        'AbKmeans_2': lambda x, y: _my_AbKmeans_segments(x, y, '2'), 
        'AbKmeans_comb': lambda x, y: _my_AbKmeans_segments(x, y, 'comb')
    }
    label_channel_2 = pseudo_gt_selector[mode_label](img, label_channel_1)
    
    #cat and return
    labels = [label_channel_0, label_channel_1, label_channel_2]
    for i, e in enumerate(labels): 
        labels[i] = e[:,:, np.newaxis] if len(e.shape) == 2 else e

    return np.concatenate(labels, axis=2)

def gen_pseudo_labels_CoNSeP(img, mask, mode_label):
    '''
    generate label data for training
    
    Parameters
    ----------
    @img: input image
    @mask: TNBC mask img
    
    Parameters
    ----------
    @A 4 channel image
    @channel 0, store ground truth
    @channel 1, store centroid and voronoi boundary
    @channel 2, store selected pseudo labels
    '''
    #find contours of each cell
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnts = []      
                                                        
    label_channel_0 = np.zeros((1000, 1000), dtype=np.uint8)     #channel 0, store ground truth
    label_channel_1 = np.zeros((1000, 1000), dtype=np.uint8)     #channel 1, store centroid and voronoi boundary
    label_channel_2 = np.zeros((1000, 1000, 2), dtype=np.uint8)  if mode_label in settings.AbKmeans_family \
                        else  np.zeros((1000, 1000), dtype=np.uint8)
                                                               #channel 2, store selected pseudo labels
                                                               
    subdiv = cv.Subdiv2D((0, 0, 1000, 1000))           #store voronoi diagram

    #draw voronoi boundary  channel_1
    for e in contours:
        ctd = _get_centroid_2(e)
        label_channel_1[ctd[1], ctd[0]] = 1
        subdiv.insert(ctd)    
    
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for e in facets:
        cv.polylines(label_channel_1, [e.astype(np.int)], True, 2)
    
    #draw ground truth    channel_0
    label_channel_0 = np.clip(mask, 0, 1)

    #draw pseudo ground truth  channel_2
    pseudo_gt_selector = {
        'Sp': lambda x, y: _my_superpixel_segments(x, y, 'CoNSeP'), 
        'Kmeans': lambda x, y: _my_kmeans_segments(x, y), 
        'AbKmeans_05': lambda x, y: _my_AbKmeans_segments(x, y, '05'), 
        'AbKmeans_1': lambda x, y: _my_AbKmeans_segments(x, y, '1'), 
        'AbKmeans_2': lambda x, y: _my_AbKmeans_segments(x, y, '2'), 
        'AbKmeans_comb': lambda x, y: _my_AbKmeans_segments(x, y, 'comb')
    }
    label_channel_2 = pseudo_gt_selector[mode_label](img, label_channel_1)
    
    #cat and return
    labels = [label_channel_0, label_channel_1, label_channel_2]
    for i, e in enumerate(labels): 
        labels[i] = e[:,:, np.newaxis] if len(e.shape) == 2 else e

    return np.concatenate(labels, axis=2)
    

def gen_pseudo_labels_TNBC(img, mask, mode_label, random_shift):
    '''
    generate label data for training
    
    Parameters
    ----------
    @img: input image
    @mask: TNBC mask img
    
    Parameters
    ----------
    @A 4 channel image
    @channel 0, store ground truth
    @channel 1, store centroid and voronoi boundary
    @channel 2, store selected pseudo labels
    '''
    #find contours of each cell
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnts = []      
                                                        
    label_channel_0 = np.zeros((512, 512), dtype=np.uint8)     #channel 0, store ground truth
    label_channel_1 = np.zeros((512, 512), dtype=np.uint8)     #channel 1, store centroid and voronoi boundary
    label_channel_2 = np.zeros((512, 512, 2), dtype=np.uint8)  if mode_label in settings.AbKmeans_family \
                        else  np.zeros((512, 512), dtype=np.uint8)
                                                               #channel 2, store selected pseudo labels
                                                               
    subdiv = cv.Subdiv2D((0, 0, 512, 512))           #store voronoi diagram

    #draw voronoi boundary  channel_1
    for e in contours:
        ctd = _get_centroid_2(e)
        if random_shift > 0:
            ctd = _add_random_shift(ctd, random_shift, 512)
        label_channel_1[ctd[1], ctd[0]] = 1
        subdiv.insert(ctd)    
    
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for e in facets:
        cv.polylines(label_channel_1, [e.astype(np.int)], True, 2)
    
    #draw ground truth    channel_0
    label_channel_0 = np.clip(mask, 0, 1)

    #draw pseudo ground truth  channel_2
    pseudo_gt_selector = {
        'Sp': lambda x, y: _my_superpixel_segments(x, y, 'TNBC'), 
        'Kmeans': lambda x, y: _my_kmeans_segments(x, y), 
        'AbKmeans_05': lambda x, y: _my_AbKmeans_segments(x, y, '05'), 
        'AbKmeans_1': lambda x, y: _my_AbKmeans_segments(x, y, '1'), 
        'AbKmeans_2': lambda x, y: _my_AbKmeans_segments(x, y, '2'), 
        'AbKmeans_comb': lambda x, y: _my_AbKmeans_segments(x, y, 'comb')
    }
    label_channel_2 = pseudo_gt_selector[mode_label](img, label_channel_1)
    
    #cat and return
    labels = [label_channel_0, label_channel_1, label_channel_2]
    for i, e in enumerate(labels): 
        labels[i] = e[:,:, np.newaxis] if len(e.shape) == 2 else e

    return np.concatenate(labels, axis=2)

def gen_pseudo_labels_CMP17(img, mask, mode_label, random_shift):
    '''
    generate label data for training
    
    Parameters
    ----------
    @img: input image
    @mask: CMP17 mask img
    
    Parameters
    ----------
    @A 4 channel image
    @channel 0, store ground truth
    @channel 1, store centroid and voronoi boundary
    @channel 2, store selected pseudo labels
    '''
    #find contours of each cell
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnts = []      
                                                        
    label_channel_0 = np.zeros((512, 512), dtype=np.uint8)     #channel 0, store ground truth
    label_channel_1 = np.zeros((512, 512), dtype=np.uint8)     #channel 1, store centroid and voronoi boundary
    label_channel_2 = np.zeros((512, 512, 2), dtype=np.uint8)  if mode_label in settings.AbKmeans_family \
                        else  np.zeros((512, 512), dtype=np.uint8)
                                                               #channel 2, store selected pseudo labels
                                                               
    subdiv = cv.Subdiv2D((0, 0, 512, 512))           #store voronoi diagram

    #draw voronoi boundary  channel_1
    for e in contours:
        ctd = _get_centroid_2(e)
        if random_shift > 0:
            ctd = _add_random_shift(ctd, random_shift, 512)
        label_channel_1[ctd[1], ctd[0]] = 1
        subdiv.insert(ctd)    
    
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for e in facets:
        cv.polylines(label_channel_1, [e.astype(np.int)], True, 2)
    
    #draw ground truth    channel_0
    label_channel_0 = np.clip(mask, 0, 1)

    #draw pseudo ground truth  channel_2
    pseudo_gt_selector = {
        'Sp': lambda x, y: _my_superpixel_segments(x, y, 'CMP17'), 
        'Kmeans': lambda x, y: _my_kmeans_segments(x, y), 
        'AbKmeans_05': lambda x, y: _my_AbKmeans_segments(x, y, '05'), 
        'AbKmeans_1': lambda x, y: _my_AbKmeans_segments(x, y, '1'), 
        'AbKmeans_2': lambda x, y: _my_AbKmeans_segments(x, y, '2'), 
        'AbKmeans_comb': lambda x, y: _my_AbKmeans_segments(x, y, 'comb')
    }
    label_channel_2 = pseudo_gt_selector[mode_label](img, label_channel_1)
    
    #cat and return
    labels = [label_channel_0, label_channel_1, label_channel_2]
    for i, e in enumerate(labels): 
        labels[i] = e[:,:, np.newaxis] if len(e.shape) == 2 else e

    return np.concatenate(labels, axis=2)


def gen_confusion_matrix(pred, pre_pseudo_label):
    '''
    class 0: background
    class 1: foreground
    '''
    
    H, W = pre_pseudo_label.shape

    if pred.shape[0] == 1:
        pred = np.concatenate((1-pred, pred), axis=0)

    y_cf = pruning.get_noise_indices(
            s=pre_pseudo_label.flatten(), 
            psx=pred.reshape(2, H*W).T
        )

    y_cf = y_cf.reshape(H, W)

    return y_cf


def gen_refined_label(pre_pseudo_label, y_cf):
    refined_label = np.copy(pre_pseudo_label)
    refined_label[(pre_pseudo_label == 0) & (y_cf==1)] += 1
    refined_label[(pre_pseudo_label == 1) & (y_cf==1)] -= 1
    
    return refined_label


def to_npimg(torch_img):
    return np.moveaxis(torch_img.to('cpu').numpy(), 0, 2)
    
    
def my_closing(img, kernel):
    if kernel is not None:
        if kernel == 'defualt':
            img = morphology.closing(img)
        elif type(kernel).__module__ == np.__name__:
            img = morphology.closing(img, kernel)
        else:
            raise ValueError('kernel should be an numpy array or pre-defined name')
            
    return img


def unit_norm(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    x = x/255
    
    return x, y

    
def min_max_norm(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    for i, e in enumerate(x):
        x[i] = (e - np.min(e)) / (np.max(e) - np.min(e))
        
    return x, y
    

def z_mean_norm(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    for i, e in enumerate(x):
        x[i] = (e - np.mean(e)) / np.std(e)
    
    return x, y
  

def get_preprocess_method(method):
    method_selector = {
        'unit': unit_norm,
        'min_max': min_max_norm,
        'z_mean':  z_mean_norm
    }
    
    return method_selector[method]
    
    
def ch_first(data):
    '''
    channel first
    '''
    new_data = []
    for e in data:
        new_data.append(np.moveaxis(e, 3, 1))
        
    return new_data


def _gkern1(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1, dtype=np.float32)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return np.array(kern2d/kern2d.sum(), dtype=np.float32)

def _gkern2(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((size, size))
    return out_filter
    
def rsv(mask, hsize=64, sigma=40, iters=9):
    eps = 1e-5
    kernel = torch.from_numpy(_gkern2(hsize, sigma)).unsqueeze(0).unsqueeze(0)
    #print(kernel.shape)
 
    for i in range(iters):
        print(i)
        M_w_pre = 0 if i == 0 else M_w
        M_bar = 1 - mask + M_w_pre

        M_w = F.conv2d(M_bar, kernel, stride=1, padding='same')
        M_w = M_w * mask

    M_w = M_w_pre / (M_w+eps)
    return M_w.numpy()
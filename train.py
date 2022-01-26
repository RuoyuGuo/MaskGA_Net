'''
Training nuclei segmentation network with only point anntation
Using our MaskGA-Net
'''

import os
from copy import deepcopy

import click
import torch
from dotmap import DotMap

from training import training_loop

def setup_data_train_kwargs(  
    #For more details, check utils.*

    #dataset
    dataset         = None,     #dataset name
    k_fold          = None,     #validation fold of 10 fold cross validation
    random_shift    = None,     #randomly move point annotation
    mode_label      = None,     #weakly labels type
    cut_size        = None,     #if split image to save memory
    data_norm       = None,     #how to normalise input images
    seed            = None,     #seed of augmentation, and dataset split

    color_norm      = None,     #color normalisation before training

    #Framework
    stage           = None,     #CL iteration times
    model_name      = None,     #directory name to save model
    framework       = None,     #framework used to train, for ablation study
    use_cl          = None,     #if use confident learning
    full_sup        = None,     #if use full supervision, ablation study
    run_pipe        = None,     #if run in pipeline, from stage = 0 to stage = YOURVALUE
    mode_stage2     = None,     #how to train network after apply CL

    #Network architecture
    cons_net_n      = None,     #consNet output channels
    is_1000         = None,     #using pad and crop in network to preserve image shape
  
    #Training
    bs              = None,     #batch size
    epochs          = None,     #training epochs
    cp_time         = None,     #checkpoint times
    attlossw_p      = None,     #weight on point labels in attention loss
    attlossw_v      = None,     #weight on voronoi labels in attention loss
    seglossw        = None,     #weight on seg loss
    attlossw        = None,     #weight on att loss
    conslossw       = None,     #weight on cons loss
    my_lr           = None,     #learning rate
    weight_decay    = None,     #weight decay value
    my_patience     = None,     #patience value of learning rate scheduler
    device          = None      #Training Device
    ):

    #weight check
    # if framework != 'SAC':
    #     seglossw = 1.0
    #     attlossw = 0.8

    #determin use_cl
    #full supervision check
    if full_sup is True:
        if use_cl is True:
            print('Training on full supervision doesnot support CL, doesnt mean you cannot use it, our code dont support this')
            print('[use_cl] is set to False.')
            print()
            use_cl = False
        if random_shift > 0:
            print('Cannot [random_shift] on full supervision.')
            print('[random_shift] is set to 0')
            print()
            random_shift = 0

    #determin use_cl
    if use_cl is True and framework == 'PseudoEdgeNet':
        print('You are using PseudoEdgeNet framework and CL, but our code dont support it.')
        print('[use_cl] is set to False.')
        print()
        usel_cl = False

    #stage check
    if use_cl is True \
        and stage < 1:
        print('You are using CL but [stage] is less than 1. [stage] is set to 1. (Ignore this message if you just want to use CL and dont want to touch [stage])')
        print()
        stage = 1
    
    if use_cl is False\
        and stage > 0:
        print('You are not using CL but [stage] is greater than 0. [stage] is set to 0. (Ignore this message if you just want to use CL and dont want to touch [stage])')
        print()
        stage = 0


    #model_name check
    if full_sup is True:
        model_name_list = [model_name, str(framework),'cs'+str(cut_size), 'N'+str(color_norm), 'full'] 
    else:
        model_name_list = [model_name, str(framework),'cs'+str(cut_size), 'N'+str(color_norm), str(mode_label), 'r'+str(random_shift)] 
   
    model_name = '-'.join(model_name_list)
    start_stage = 0

    run_pipe = False
    if use_cl is True:
        for i in range(stage-1, -1, -1):
            network_b_path = os.path.join('.', 'checkpoints', dataset, 'stage'+str(i),  model_name , str(k_fold)+'.pt')
            network_f_path = os.path.join('.', 'checkpoints',  dataset, 'stage'+str(i), model_name, str(k_fold)+'_f.pt')
    
            if os.path.isfile(network_f_path) is False or os.path.isfile(network_b_path) is False :
                run_pipe = True
                start_stage = i
                
        if run_pipe is True:
            print(f'You are using CL, but it looks like you didnt finish to train a network with stage={stage-1}')
            print(f'Program will train a newtork on previous stages with the same setting first. Then use CL')
            print(f'If you wish to use different settings, see train.py --help or README.md')
            print()

    #is_1000 check
    if cut_size == 0 and dataset == 'MoNuSeg':
        print('0 pad and crop in Network is active. (Ignore this message if you dont want to touch network architecture)')
        print()
        is_1000 = True

    #device check
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #new parameters
    #run_pipe, is_1000, network_path

    #used for my_dataset.py
    cfgs = {
        'stage'             : stage,
        'cp_time'           : cp_time,
        'dataset'           : dataset,
        'k_fold'            : k_fold,
        'random_shift'      : random_shift, 
        'model_name'        : model_name,
        'mode_label'        : mode_label,
        'cut_size'          : cut_size,
        'bs'                : bs,
        'seed'              : seed,
        'data_norm'         : data_norm,
        'color_norm'        : color_norm,
        'epochs'            : epochs,
        'attlossw_p'        : attlossw_p,      
        'attlossw_v'        : attlossw_v,    
        'seglossw'          : seglossw,       
        'attlossw'          : attlossw,     
        'conslossw'         : conslossw,    
        'my_lr'             : my_lr,
        'weight_decay'      : weight_decay,
        'my_patience'       : my_patience,
        'mode_stage2'       : mode_stage2,
        'cons_net_n'        : cons_net_n,
        'framework'         : framework, 
        'use_cl'            : use_cl,
        'is_1000'           : is_1000,
        'full_sup'          : full_sup,
        'run_pipe'          : run_pipe,
        'device'            : device,
        'start_stage'       : start_stage
    }

    return DotMap(cfgs, _dynamic=False)


def running(cfgs, cur_stage):
    '''
    running for one stage
    '''
    #parameters are fixed from here
    #create copy of current config
    cur_cfgs = deepcopy(cfgs)
    cur_cfgs.stage = cur_stage

    #modify current config
    if cur_stage == 0:
        cur_cfgs.use_cl = False
    else:
        cur_cfgs.use_cl = True

    training_loop.train(cur_cfgs)


def running_in_pipe(cfgs):
    #run from stage=0 to stage=N
    for i in range(cfgs.start_stage, cfgs.stage+1):
        if i == 0:
            print(f'Framework will run without CL [stage={i}]')
        else:
            print(f'Framework will run with CL [stage={i}]')

        print()
        running(cfgs, cur_stage=i)


@click.command()

#dataset options
@click.option('--dataset', required=True, \
                help='Training on which dataset')
@click.option('--k_fold', type=click.IntRange(0, 9), default=0, \
                help='Use which Testset fold in 10 cross validation')
@click.option('--model_name', required=True, \
                help='Folder name when save result, keep consistency if you use_Cl, it is recommended to use unique name for different settings')
@click.option('--framework', type=click.Choice(['Seg', 'Seg_Att', 'SAC', 'PseudoEdgeNet']), \
                default='SAC', \
                help='Which training framework is used for ablation study, performance: SAC>Seg_Att>PseudoEdgeNet>Seg')
@click.option('--stage', type=click.IntRange(0), default=1, \
                help='Iteration times to apply CL, only work when use_Cl, use this para only if you are familiar with CL, otherwise only use [use_cl]')
@click.option('--cp_time', type=click.Choice(['best', 'final']), default='best', \
                help='Checkpoint of network.\nbest:best performance on validation set.\nfinal:network after training last epoch')
@click.option('--random_shift', type=click.IntRange(0), default=0, \
                help='Move range(in pixel), randomly shift each point annotation')
@click.option('--mode_label', type=click.Choice(['Sp', 'Kmeans', 'AbKmeans_05', 'AbKmeans_1', 'AbKmeans_2', 'AbKmeans_comb']), \
                default='Sp', \
                help='Type of pseudo label to train')
@click.option('--cut_size', type=click.IntRange(0), default=0, \
                help='Using divided image, should use supervision_tools.py to prepare data')
@click.option('--bs', type=click.IntRange(1), default=2, \
                help='Batch size')
@click.option('--seed', type=click.IntRange(0), default=12, \
                help='Seed for data augmentation, dataset splitted')
@click.option('--data_norm', type=click.Choice(['unit', 'min_max', 'z_mean']) , default='unit' , \
                help='How to normalise input images, set to None to avoid normalisation')
@click.option('--color_norm', default=None, type=click.Choice(['norm1']), \
                help='Color normalisation method on input images')

#training options
@click.option('--use_cl', type=click.BOOL, default=True, \
                help='Applying confident learning')
@click.option('--full_sup', type=click.BOOL, default=False, \
                help='Using full supervision rather than point annotations (weakly)')
@click.option('--epochs', type=click.IntRange(1), default=60, \
                help='Training epochs')
@click.option('--attlossw_p', type=click.FloatRange(0), default=1, \
                help='Weight of point labels in att loss')
@click.option('--attlossw_v', type=click.FloatRange(0), default=0.4, \
                help='Weight of voronoi labels in att loss')
@click.option('--seglossw', type=click.FloatRange(0), default=0.8, \
                help='Weight of seg loss')
@click.option('--attlossw', type=click.FloatRange(0), default=1, \
                help='Weight of att loss')
@click.option('--conslossw', type=click.FloatRange(0), default=0.4, \
                help='Weight of cons loss')
@click.option('--my_lr', type=click.FloatRange(0), default=0.001, \
                help='Learning rate value')
@click.option('--weight_decay', type=click.FloatRange(0), default=0.0005, \
                help='Weight decay value')
@click.option('--my_patience', type=click.IntRange(1), default=4, \
                help='Patience of ReduceLROnPlateau in Pytorch')
@click.option('--mode_stage2', type=click.Choice(['retrain', 'refine']), default='retrain', \
                help='After correct the pseudo label, how to use them to train network')
@click.option('--cons_net_n', type=click.IntRange(1), default=4, \
                help='Output channel of ConsNet')

def main(**kwargs):
    """'Training MaskGA_Net on MoNuSeg, TNBC, CPM17 and your own dataset
    '

    #########################################################
    Examples (use same settings before/after CL):

    \b
    #Training on MoNuSeg without CL
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME 
    
    \b
    #Training on MoNuSeg with CL
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl True

    \b
    #Training on TNBC without CL
    python train.py --dataset TNBC --model_name YOURMODELNAME 

    \b
    #Training on TNBC with CL
    python train.py --dataset TNBC --model_name YOURMODELNAME --use_cl True

    #########################################################
    Examples (use different settings before/after CL):

    \b
    run 
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --attlossw 0.2
    then run
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl True --attlossw 0.8

    #########################################################
    Examples (Iteratively using CL, use same settings before/after CL, N>=2)

    \b
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl True --stage N

    #########################################################
    Examples (Iteratively using CL, use different settings before/after CL)

    \b
    run
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --attlossw 0.2
    then
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl True --attlossw 0.8
    then 
    python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl True --stage 2 --attlossw 0.4
    so on...


    #########################################################
    \b
    Best performance:
    --framework SAC --use_cl True
    """


    #setup arguments
    cfgs = setup_data_train_kwargs(**kwargs)

    # if running from stage=0 to stage=N>0
    if cfgs.run_pipe is True:
        print('Framework will continually run from withou CL to with CL...')
        print()
        running_in_pipe(cfgs)
    else:
        print('Framework will start to run...')
        print()
        running(cfgs, cur_stage=cfgs.stage)


if __name__ == '__main__':
    main()
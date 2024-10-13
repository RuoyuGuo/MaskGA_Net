## Learning with Noise: Mask-Guided Attention Model for Weakly Supervised Nuclei Segmentation (MICCAI 2021)
## SAC-Net: Learning with weak and noisy labels in histopathology image segmentation (MedIA 2023)
---
Offical PyTorch implementation

## Updates
---

**26-01-2021**: Code release, support MoNuSeg, TNBC, CPM17 dataset 

**XX-10-2021**: We are refining the code and preparing the paper for another new journal.  We also wish to provide code that can works on your own dataset. The code will be public after the paper submitted. Thank you for your patience.

**XX-09-2021**: Code should be pushed at the end of October.

## Repository structure
---
| Path | Description
| :--- | :---
| MaskGA_Net | Main directory
| &ensp;&ensp;&boxvr;&nbsp; train.py | start your training
| &ensp;&ensp;&boxvr;&nbsp; supervision_tools.py | Generate different types of pseudo labels **before** training
| &ensp;&ensp;&boxvr;&nbsp; requirements.txt | environment
| &ensp;&ensp;&boxvr;&nbsp; README.md | document
| &ensp;&ensp;&boxvr;&nbsp; training | training proceduro code
| &ensp;&ensp;&boxvr;&nbsp; network | network code
| &ensp;&ensp;&boxvr;&nbsp; loss | loss function, evaluation metrics code
| &ensp;&ensp;&boxvr;&nbsp; metrics | evaluation metrics function
| &ensp;&ensp;&boxvr;&nbsp; utils | other code
| &ensp;&ensp;&boxvr;&nbsp; img_norm | image preprocessing code
| &ensp;&ensp;&boxur;&nbsp; datasets | put dataset in here
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; MoNuSeg | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; original | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Input_Images | input images
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Labels_GT_XML | XML ground truth file 
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; TNBC | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; original | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Input_Images | input images
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Labels_GT_PNG | ground truth images
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; CPM17 | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; original | 
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Input_Images | input images
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; Labels_GT_PNG | ground truth images

## Preparing
---
1. install libraries `pip install -r requirements.txt`
1. download [MoNuSeg](https://nucleisegmentationbenchmark.weebly.com/), [TNBC](https://ieee-dataport.org/documents/segmentation-nuclei-histopathology-images-deep-regression-distance-map), [CPM17](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) dataset
2. ***MoNuSeg***:
    1. Put every and only input image files (*.png) in ```./datasets/original/MoNuSeg/Input_Images```
    2. Put every and only ground truth files (*.xml) in
 ```./datasets/original/MoNuSeg/Labels_GT_XML```
3. ***TNBC***:
    1. TNBC original dataset splits images into different folds, you only need to put these images into the same fold
    2. Put every and only input image files (*.png) in ```./datasets/TNBC/Input_Images```
    3. Put every and only ground truth images files (*.png) in ```./datasets/TNBC/Labels_GT_PNG```
3. ***CPM17***:
    1. The CPM17 dataset provided by HoVer-Net includes two folder, `train` and `test`, We use both to train. So put images in both `train` and `test` in ```./datasets/original/CPM17/Input_Images```.
    2. The ground truth files are in `.mat` format. Please follow the [HoVer-Net](https://github.com/vqdang/hover_net) to convert them to image files, and make sure these image files are greyscale with only two values (255 denotes the foreground, and 0 denotes the background). Then put these image files in ```./datasets/original/CPM17/Labels_GT_PNG```

## Generating pseudo labels
Before start your training, you need to generate some pseudo labels. Some arguments in `train.py` specify the detail of pseudo label used for training, these details should be equal to the pseudo label you just generate by running `supervision_tools.py`. Here are some examples. There are lots of different types of pseudo labels with different influences on results, some are good, some are bad. You may want to try by yourself.

If you want to train TNBC with superpixels labels, run
`python supervision_tools.py --dataset TNBC` first. 

If you want to train MoNuSeg with superpixels labels, but randomly move points within 5 pixels, run `python supervision_tools.py --dataset MoNuSeg --random_shift 5` first.

Some images might be too large to fit into low memory GPU, to solve this problem, you can divide each images into smaller part, like divide MoNuSeg into 4 parts, each parts with 512 X 512 dimensions.  You can run run `python supervision_tools.py --dataset MoNuSeg --cut_size 512`

## Quick start (Training in one line code)
---
Training results will be saved in `./checkpoints`

```
# Generating pseudo labels
python supervision_tools.py --dataset MoNuSeg
python supervision_tools.py --dataset TNBC
python supervision_tools.py --dataset CPM17

# Training on MoNuSeg without CL
python train.py --dataset MoNuSeg --model_name YOURMODELNAME --use_cl False

# Training on MoNuSeg with CL
python train.py --dataset MoNuSeg --model_name YOURMODELNAME 

# Training on TNBC without CL
python train.py --dataset TNBC --model_name YOURMODELNAME --use_cl False

# Training on TNBC with CL
python train.py --dataset TNBC --model_name YOURMODELNAME

# Training on CPM17 without CL 
python train.py --dataset CPM17 --model_name YOURMODELNAME --use_cl False

# Training on CPM17 with CL 
python train.py --dataset CPM17 --model_name YOURMODELNAME 
```

## Using different settings before/after CL
---
The above with CL exmaples will run on previous stages with the same settings first, then run with CL, if you dont training on without CL before. However, using different settings before/after CL may affect final results, our project also support this.  For exmaple, if you want to use different weight on attention loss, before/after CL.

It is important to use the same `model_name` when you use different settings since it requires previous network to denoise. Therefore, in the following three lines code, `YOURMODELNAME` in the first line `==` `YOURMODELNAME` in the second line `==` `YOURMODELNAME` in the third line.

run 

`python train.py --data_name MoNuSeg --model_name YOURMODELNAME --attlossw 0.2`

then run

`python train.py --data_name MoNuSeg --model_name YOURMODELNAME --use_cl True --attlossw 0.8`

if you want to keep going to use different weight on attention loss on `stage=2`, 

then run

`python train.py --data_name MoNuSeg --model_name YOURMODELNAME --use_cl True --stage 2 --attlossw 0.4`

## Iteratively train your network
---
In `train.py`, `stage` tells the framework how many times to iteratively training networks, also use [confident learning(CL)](https://github.com/cleanlab/cleanlab) to revise the pseudo ground truth for the next iteration. For exmaples, when `stage=0`, it will directly train networks and stop. When `stage=1`, it will use the information from newtorks trained by`stage=0` to train new networks, and so on so forth. 

`use_cl` works similarly with `stage`, and it tell the network if apply CL to revise pseudo ground. `use_cl` has a higher priority than `stage`. This parameter give a more convenient way to use code than `stage`, even you don't know how the project works. If you don't want to go in deeper in CL, set `use_cl` and ignore `stage`. By setting `use_cl=True` or `use_cl=False`, you could train network with or without revised pseudo ground truth (Also you can ignore any notifications when you run `train.py`)

However, you may want to see if you could generate better result by applying CL multiple times, say 5. To do so, you can set `use_cl=True, stage=5`.

## Citations and Related Publications
---
If you find our code useful, please consider to cite our paper.
```
@InProceedings{10.1007/978-3-030-87196-3_43,
    author={Guo, Ruoyu and Xie, Kunzi and Pagnucco, Maurice and Song, Yang},
    title={Learning with Noise: Mask-Guided Attention Model for Weakly Supervised Nuclei Segmentation},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    year={2021},
    pages={461--470},
}

@article{guo2023sac,
  title={SAC-Net: Learning with weak and noisy labels in histopathology image segmentation},
  author={Guo, Ruoyu and Xie, Kunzi and Pagnucco, Maurice and Song, Yang},
  journal={Medical Image Analysis},
  volume={86},
  pages={102790},
  year={2023},
}
```

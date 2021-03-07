


This directory contains the code for training, evaluating our FastDepth model for image semantic segmentation. 

 

## Setup

1. Clone repository:
    ```bash
    git clone https://github.com/idonahum/FastDepth.git
   
    cd /path/to/this/directory
    ```

2. Set up anaconda environment including all dependencies:
    ```bash
    # create conda environment from YAML file
    conda env create -f rgbd_segmentation.yaml
    # activate environment
    conda activate rgbd_segmentation
    ```

## 

    The folder [`preprocess/datasets`](preprocess/datasets) contains the code to prepare
    NYUv2 for training and evaluation. 
    Please follow the instructions given for the respective dataset and store 
  

## Content
There are subsection for different things to do:
- [Trained Models](#trained_models)
- [Training](#training)
- [Evaluation](#evaluation)

## Trained Models ##
  We provide one trained model based on mobilenetv1 as a backbone.
  - MobileNet-NNConv5(depthwise) - filename FastSeg_l1_mobilenet.pth and it is available in Weights directory. 

## Training ##
to train a new model. run the following command:
```bash
python3 segmentation.py -mode train -backbone [encoder_type] --gpu True
```
change encoder_type to one of the following: [mobilenet, mobilenetv2]


all checkpoints save to the path - FastDepth/Weights/[backbone]/[criterion]/

please make sure that the directory exists before running train. for example, if you would like to use mobilenet as a backbonen, make sure that the directory 'FastDepth/Weights/mobilenet/ce ' - exists.

if you want to resume your training, run the following command:
```bash
python3 segmentation.py -mode train -backbone [encoder_type] --gpu True --resume [path_to_checkpoint]
```
### Pretrained MobileNet ###

-The model file for the pretrained MobileNet used in our model definition can be downloaded from [http://datasets.lids.mit.edu/fastdepth/imagenet/](http://datasets.lids.mit.edu/fastdepth/imagenet/).
when downloading it, make sure to put the tar file in 'Weights' directory
-We provide in 'Weights' directory, the pretrained weights for MobilNetV2.
## Evaluation ##

to evalaute an existing model, weights should be inside the 'Weights' directory. use the following command to run evaluation.

```bash
python3 segmentation.py -mode eval -backbone [mobilenet or mobilenetv2] --pretrained [model_weights_filename] --gpu True
```
if you would like to use our pretrained weights:
```bash
python3 segmentation.py -mode eval -backbone mobilenet --pretrained FastSeg_l1_mobilenet.pth --gpu True
```
The evaluation code will report model accuracy in terms of mIou[%].



## Results

All results avaiable here: [link to pdf](https://drive.google.com/file/d/1Ws-wl00jpNyFQzDz-7gIQf0yQPuOQLur/view?usp=sharing)

### Reference 
our code for semantic segmentation data preprocessing is highly depends on the following paper.

>Seichter, D., Köhler, M., Lewandowski, B., Wengefeld T., Gross, H.-M.
*Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis*
arXiv preprint arXiv:2011.06961 (2020).

```bibtex
@article{esanet2020,
  title={Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis},
  author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
  journal={arXiv preprint arXiv:2011.06961},
  year={2020}
}
```

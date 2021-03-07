This directory provides insturctions how to train, test models for semantic segmentation.


This directory contains the code for training, evaluating our FastDepth model for image semantic segmentation. 

 

## Setup

1. Clone repository:
    ```bash
    git clone https://github.com/TUI-NICR/ESANet.git
   
    cd /path/to/this/repository
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

4. Pretrained models (evaluation):  
  

## Content
There are subsection for different things to do:
- [Evaluation](#evaluation): Reproduce results reported in our paper.
- [Dataset Inference](#dataset-inference): Apply trained model to samples from dataset.
- [Sample Inference](#sample-inference): Apply trained model to samples in [`./samples`](samples).
- [Time Inference](#time-inference): Time inference on NVIDIA Jetson AGX Xavier using TensorRT.
- [Training](#training): Train new ESANet model.

## Evaluation

Examples: 
- To evaluate our ESANet-R34-NBt1D trained on NYUv2, run:
    ```bash
    python eval.py \
        --dataset nyuv2 \
        --dataset_dir ./datasets/nyuv2 \
        --ckpt_path ./trained_models/nyuv2/r34_NBt1D.pth
     
    # Camera: kv1 mIoU: 50.30
    # All Cameras, mIoU: 50.30
    ```
- To evaluate our ESANet-R34-NBt1D trained on SUNRGB-D, run:
    ```bash
    python eval.py \
        --dataset sunrgbd \
        --dataset_dir ./datasets/sunrgbd \
        --ckpt_path ./trained_models/sunrgbd/r34_NBt1D.pth
    
    # Camera: realsense mIoU: 32.42
    # Camera: kv2 mIoU: 46.28
    # Camera: kv1 mIoU: 53.39
    # Camera: xtion mIoU: 41.93
    # All Cameras, mIoU: 48.17
    ```



### Training

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

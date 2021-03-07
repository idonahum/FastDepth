FastDepth
============================

This repo is re-implentation of FastDepth project at MIT, we present up to date code, with extra trained models based on different backbones and different loss functions. we provide pretrained models. and insturctions how to train and evalute models. [FastDepth](http://fastdepth.mit.edu/) p



## Contents
0. [Requirements](#requirements)
0. [Trained Models](#trained-models)
1. [Training](#training)
2. [Evaluation](#evaluation)
4. [Results](#results)
5. [Reference](#reference)

## Requirements
- Install [PyTorch](https://pytorch.org/)
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format libraries. Files in our pre-processed datasets are in HDF5 format.
  ```bash
  sudo apt-get update
  sudo apt-get install -y libhdf5-serial-dev hdf5-tools
  pip3 install h5py matplotlib imageio scikit-image opencv-python
  ```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 format. The NYU dataset requires 32G of storage space.
	```bash
	cd FastDepth
	python3 prepare_dataset.py ../FastDepth
	```
downloading and unziping takes a while, please be patient.
## Trained Models ##
  The following trained models can be found at 
  - MobileNet-NNConv5(depthwise) with additive skip connections. Loss function: L1
  - MobileNetV2-NNConv5(depthwise) with additive skip connections. Loss function: L1
  - MobileNet-NNConv5(depthwise) with additive skip connections. Loss function: RMSLE+ Gradient+ Normal
  - MobileNetV2-NNConv5(depthwise) with additive skip connections. Loss function: RMSLE+ Gradient+ Normal
  - MobileNet-NNConv5(depthwise) with additive skip connections. Loss function: L1+ Gradient+ Normal
  - MobileNetV2-NNConv5(depthwise), with additive skip connections. Loss function: L1+ Gradient+ Normal

## Training ##
to train a new model. run the following command:
```bash
python3 main.py -mode train -backbone [encoder_type] --criterion [criterion] --gpu True
```
change encoder_type to one of the following: [mobilenet, mobilenetv2]

change the criterion you would like to use for training, option are: [l1, l1gn, rmslegn]

all checkpoints save to the path - FastDepth/Weights/[backbone]/[criterion]/

please make sure that the directory exists before running train. for example, if you would like to use mobilenet as a backbone and l1 as a criterion, make sure that the directory 'FastDepth/Weights/mobilenet/l1 ' - exists.

### Pretrained MobileNet ###

The model file for the pretrained MobileNet used in our model definition can be downloaded from [http://datasets.lids.mit.edu/fastdepth/imagenet/](http://datasets.lids.mit.edu/fastdepth/imagenet/).

## Evaluation ##

to evalaute an existing model, weights should be inside the 'Weights' directory. use the following command to run evaluation.

```bash
python3 main.py -mode eval -backbone [mobilenet or mobilenetv2] --criterion [criterion] --pretrained [model_weights_filename] --gpu True
```
change criterion to one of the following options: [l1, l1gn, rmslegn]

The evaluation code will report model accuracy in terms of the delta1 metric as well as RMSE metric.



## Results

All results avaiable here: [link to pdf]

## Referece

	@inproceedings{icra_2019_fastdepth,
		author      = {{Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne}},
		title       = {{FastDepth: Fast Monocular Depth Estimation on Embedded Systems}},
		booktitle   = {{IEEE International Conference on Robotics and Automation (ICRA)}},
		year        = {{2019}}
	}

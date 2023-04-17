# Heatmap-based Out-of-Distribution Detection

This repository contains the official implementation of our  WACV 2023 paper. 

## Requirements 
We provide the `environment.yml` file with the required packages. The file can be used to create an Anaconda environment.

## Datasets
We use CIFAR-10, CIFAR-100 and Tiny ImageNet as in-distribution datasets for the classifier pretraining. To train the heatmap decoder, we rely on the in-distribution training sets as well as on 80 Million TinyImages for CIFAR-10/CIFAR-100 and Places365 for Tiny ImageNet. 
We test the CIFAR-10/CIFAR-100 models on the respective in-distribution test set as well as iSUN, LSUN-Crop, LSUN-Resize, SVHN, Textures and Places365. We test the Tiny ImageNet model on iNaturalist, SUN and Textures. 
All datasets should be placed in the `Datasets` folder. 


## Pre-trained Models 
The pre-trained classification models can be downloaded here: [Models](https://cloudstore.uni-ulm.de/s/Wtpcebb3i4NXDcp). The models should be places in the folder `Models`.

## Run Code 
The classifiers can either be trained with the file `classifier_pretraining.py` or downloaded as described above. The device can be changed with the argument `-device`. We use `-device cuda:0` as default.

### Classifier Pre-training 
We use `ResNet18` or `WideResNet` wit depth `40` and width `2` as classifiers for CIFAR-10/CIFAR-100 and `ResNet50` as classifier for Tiny ImageNet.
The classifiers can be pre-trained by running the following command: 

```
python3 classifier_pretraining.py -dataset cifar10 -arch resnet18 
```

### Heatmap Decoder Training 
The configuration files for the heatmap decoder training are placed in the folder `configs`. We provide a configuration file for each of the five settings. 
The heatmap decoder can be trained by the following command: 

```
python3 ood_training.py -config ./configs/cifar10_resnet18_ood.json 
```

The number of used out-of-distribution training samples can be changed with the argument `-num_ood`. 
The trained heatmap decoder is then placed in a new directory in the folder `checkpoints`. The directory is named with the in-distribution dataset, the classifier architecture and a timestamp. 

### Out-of-Distribution Evaluation 
The trained heatmap decoder can be evaluated with the following command: 

```
python3 eval_ood.py -config ./configs/cifar10_resnet18_ood.json -checkpoint_name cifar10_resnet18_ood_fw1_2022_07_05_07_27_20
```

If no checkpoint folder is specified with the argument `checkpoint_name`, the last checkpoint created with the corresponding configuration is selected. 

## Reference
Please use the following citations when referencing our work:

**Heatmap-based Out-of-Distribution Detection** \
*Julia Hornauer and Vasileios Belagiannis* **[[paper]](https://arxiv.org/abs/2211.08115)**
```
@article{Hornauer2022HeatmapbasedOD,
  title={Heatmap-based Out-of-Distribution Detection},
  author={Julia Hornauer and Vasileios Belagiannis},
  journal={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2022},
  pages={2602-2611}
}
``` 
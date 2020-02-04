# Peking University/Baidu - Autonomous Driving
This repo is the source code of my solution in [Peking University/Baidu - Autonomous Driving](https://www.kaggle.com/c/pku-autonomous-driving/). <br />
The main architecture of this solution is CenterNet, which is a very great work from [Objects as Points](https://arxiv.org/abs/1904.07850). <br />
<br />
<br />
## Overview
<br />
The main task of this competition is to detect the cars on the street. The detection task is not just on 2D coordinate, but 3D world coordinate. <br />
So the ground truths of this competition are the 3D rotation angles(Pitch, Yaw and Roll) and world coordinate corresponding to the data collection camera. <br />
With these information, we can know the distance between cars, and how the cars are going to turn. <br />
<br />
<br />

## Data augmentation

<br />

**1. Random Gamma** <br />
**2. Random brightness** <br />
**3. Random contrast** <br />
**4. Gaussian noise** <br />
**5. Blur** <br />
**4. Horizontal flip** <br />
**5. Camera rotation** <br />

<br />

Camera rotation is added after the comptition, this is a very important augmentation techique in this competition. <br />
And this augmentation techique is shared by a person who won the first place - [outrunner](https://www.kaggle.com/outrunner) in this compeititon. <br />
Since there are only around 4000 pictures for training, so camera rotation is a great way to expand the training dataset. <br />

## Model architecture

I trained 4 models for this competition, and ensemble them in the end to be my final submission answer. <br />
There are:<br />
<br />

**1. CenterNet(UNet++ decoder) with efficietnet B3 backbone** <br />
**2. CenterNet(msra decoder) with efficietnet B2 backbone** <br />
**3. CenterNet(msra decoder) with efficietnet B3 backbone** <br />
**4. CenterNet(msra decoder) with efficietnet B4 backbone** <br />
<br />
I changed the decoder to UNet++ in the last few days. It can provide better mAP score, but the training time is also longer. <br />
All the models have the same prediction heads, there are : <br />
<br />
**1. Heatmap head (for keypoint detection)** <br />
**2. Rotation head (for Yaw, Pitch and Roll regression)** <br />
**3. Depth head (for distance regression)** <br />

<br />

The implementation of Rotation head came from a great public [kernel](https://www.kaggle.com/hocop1/centernet-baseline) in this competition. <br />
And the heatmap and depth heads came from the original article of CentetNet. <br />
But there is a different between my implementation and original article, I performed the sigmoid activation on output of heatmap head. <br />
This change can make the training much stable. <br />
<br />

## Loss functions
<br />
**Focal Loss for heatmap head**
**L1 Loss for rotation and depth head**
The total loss is sum by some weightings of these 3 losses, which are `0.1 : (0.9)*1.25 : (0.9)*1.5 = heatmap_loss : rotation_loss : depth_loss` <br />
<br />

## Training recipe
<br />

**Training data : 80% of original dataset, validation data  : 20% of original dataset**. <br />

(I didn't perform k-fold training, so there is no hold-out dataset.) <br /> 

**Total training epochs : around 40-50 epochs** <br />

**Intital learning rate : 6e-4** <br />

**Optimizer : Adam optimizer** <br />

The weightings of each loss are coming from many trials of training, which I add the all the losses into metric. <br />
After I settled down the weightings, I discard the validation on losses of validation dataset. <br />
Instead, I use the prediction of validation dataset to calculate the mAP in the end of every epoch, and use the mAP score to be the monitering metric of training. <br />
The learning rate is decreased by x = x*0.5 if the validation mAP didn't improve in 2 epochs. <br />
<br />

## Demo 
<br />

![ScreenShot](demo/demo1.png)

<br />

![ScreenShot](demo/demo2.png)

<br />

## Reference

<br />
[Object of points](https://arxiv.org/abs/1904.07850) <br />
[Original CenterNet github repo](https://github.com/xingyizhou/CenterNet) <br />
[CenterNet github repo](https://github.com/xuannianz/keras-CenterNet) <br />
[CenterNet public kernel - 1](https://www.kaggle.com/hocop1/centernet-baseline) <br />
[AR public kernel - 2](https://www.kaggle.com/ebouteillon/augmented-reality) <br />
[Metrics public kernel -3](https://www.kaggle.com/its7171/metrics-evaluation-script) <br />
[3D visualization public kernel - 4](https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car) <br />
[Camera rotation public kernel - 5](https://www.kaggle.com/outrunner/rotation-augmentation) <br />
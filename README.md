# Image Processing with CNN

## CNN Architecture

 The  CNN  architecture for this  assignment  is  given  in  Figure  2.  This  network  has  two  convlayers,  and  three fc  layers.  Each  convlayer  is  followed  by  a  max  poolinglayer.

 ### Conv Layers:
 Both conv  layers  accept  an  inputreceptive field of spatial size 5x5. The filter numbers of the first and the second conv layers are 6 and 16 respectively. The stride parameter is 1 and no padding is used. 

 ### Pool Layers:
 The twomax poolinglayers take an input window  size  of  2x2,  reduce  the  window  size  to  1x1  by  choosing  the  maximum  value  of  the  four  responses. 
 
 ### FC Layers:
 The first two fc layers have 120 and 80 filters, respectively. The last fc layer, the output layer, has  size  of  10  to  match  the  number  of  object  classes  in  the  MNIST  dataset.  Use  the  popular  ReLU  activation function [3] for all convand all fc layers except for the output layer, which uses softmax [4] to compute the probabilities.

## TO DO
1. Plot epoch-accuracy
    * https://discuss.pytorch.org/t/easiest-way-to-draw-training-validation-loss/13195
    * https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/
    * https://github.com/TeamHG-Memex/tensorboard_logger
2. Find 5 parameters to change to change epoch-accuracy
    * Ideas: batchSize, epochs, number strides, etc.

## Installing Correct NVIDIA Drivers for Ubuntu 18.04 Bionic Beaver
Use the Following Link: https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux

Follow the "Manual Install using the Official Nvidia.com driver" section

NOTE: Installing nvidia drivers through apt or apt-get will result in a bootup issue! Follow the manual download process and build from source.

# Image Processing with CNN

## General CNN Instructions

 The  CNN  architecture for this  assignment  is  given  in  Figure  2.  This  network  has  two  convlayers,  and  three fc  layers.  Each  convlayer  is  followed  by  a  max  poolinglayer.

 ### Conv Layers:
 Both conv  layers  accept  an  inputreceptive field of spatial size 5x5. The filter numbers of the first and the second conv layers are 6 and 16 respectively. The stride parameter is 1 and no padding is used. 

 ### Pool Layers:
 The twomax poolinglayers take an input window  size  of  2x2,  reduce  the  window  size  to  1x1  by  choosing  the  maximum  value  of  the  four  responses. 
 
 ### FC Layers:
 The first two fc layers have 120 and 80 filters, respectively. The last fc layer, the output layer, has  size  of  10  to  match  the  number  of  object  classes  in  the  MNIST  dataset.  Use  the  popular  ReLU  activation function [3] for all convand all fc layers except for the output layer, which uses softmax [4] to compute the probabilities.


 ## Discussion 11 Notes
 layers
 self.conv1 = Conv2d(3, 6, 5)
 self.pool = MaxPool2d(2, 2)
 self.conv2 = Conv2d(6, 16, 5)
 self.fc1 = Linear(16 * 5 * 5, 120)
 self.fc2 = Linear(120, 84)
 self.fc3 = Linear(84, 10)

 forward function fwd(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(X)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

Step 1:
Step 2:
Step 3: Define a Loss Function and Optimizer
- Loss fn: nn.CrossEntropyLoss()
    * criterion(outputs, labels)
    * output against labels (cross entropy)
    
- Optimizer: optim.SGD()
    * update weight

Testing
 
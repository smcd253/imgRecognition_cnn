# -*- coding: utf-8 -*-
from collections import deque # for easy list manipulation
import torch
import torchvision
import torchvision.transforms as transforms


#####################################
# BASIC FUNCTIONS

def plotBatchAccuracy(epoch_accuracy, batchSize): # source: https://stackoverflow.com/questions/19189488/use-a-loop-to-plot-n-charts-python
    fig = plt.figure()
    ax = fig.add_subplot(111)
    epoch_accuracy.append(0)
    epoch_accuracy.rotate(1)
    xint = range(1, epoch_accuracy.__len__())
    plt.xticks(xint)
    ax.plot(epoch_accuracy)
    epoch_accuracy.popleft() # trim 0 value to get max and min
    plt.xlim(left=1)
    plt.ylim(bottom = min(epoch_accuracy), top = max(epoch_accuracy))
    fig.suptitle("Test Accuracy per Epoch (Batch Size = %d)" % (batchSize))
    fig.savefig("torchResults/testAcc_batchSize_%d.png" % (batchSize))

def plotLoss(lossesByEpoch, batchSize):
    fig = plt.figure()
    axs = fig.add_subplot(111)
    for l in lossesByEpoch:
        l.insert(0, 0)
    xint = range(1, lossesByEpoch[0].__len__())
    plt.xticks(xint)
    for epoch,losses in enumerate(lossesByEpoch):
        axs.plot(losses, label = 'epoch %d' % (epoch + 1))
    plt.xlim(left=1)
    plt.legend(loc='upper left')
    fig.suptitle("Training Loss by Epoch (Batch Size = %d)" % (batchSize))
    fig.savefig('torchResults/trainLoss_by_epoch_batchSize_%d' % (batchSize))

def plotAccuracy(trainAccByEpoch, batchSize):
    fig = plt.figure()
    axs = fig.add_subplot(111)
    for a in trainAccByEpoch:
        a.insert(0,0)
    xint = range(1, trainAccByEpoch[0].__len__())
    plt.xticks(xint)
    for epoch,accuracies in enumerate(trainAccByEpoch):
        axs.plot(accuracies, label = 'epoch %d' % (epoch + 1))
    plt.xlim(left = 1)
    plt.legend(loc='upper left')
    fig.suptitle("Training Accuracy by Epoch (Batch Size = %d)" % (batchSize))
    fig.savefig('torchResults/trainAcc_by_epoch_batchSize_%d' % (batchSize))


# train on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print("DEVICE: ")
print(device)
print()

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# we're going to loop through batch sizes of 1, 2, and 4 to test the 
# effect of different batch sizes

import matplotlib.pyplot as plt
import numpy as np

# use dictionary to keep track of multiple plots
figs = {}
figEpochAcc, axEpochAcc = plt.subplots()

batch_sizes = [1, 2, 4, 8, 16]
num_epochs = 16
for b, batchSize in enumerate(batch_sizes):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))])

    trainset = torchvision.datasets.MNIST(root='/home/spencer/Documents/ee569/MNIST', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='/home/spencer/Documents/ee569/MNIST', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                            shuffle=False, num_workers=2)

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


    ########################################################################
    # 2. Define a Convolutional Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, stride = 1, padding = 0)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5, stride = 1, padding = 0)
            self.fc1 = nn.Linear(16 * 4 * 4, 120) # x shape is 16 * 4 * 4
            self.fc2 = nn.Linear(120, 80)
            self.fc3 = nn.Linear(80, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            #print(x.shape)
            x = x.view(-1, 16 * 4 * 4) # x shape is 16 * 4 * 4
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    #transfer to GPU
    net.to(device)

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    net.train() # sets model to training mode (if you have drouout and BN, Batch Nom)

    lossesByEpoch = []
    trainAccByEpoch = []
    epoch_accuracy = deque([]) # list to hold accuracy of test over epochs

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        losses = []
        accuracies = []
        running_loss = 0.0
        correct_t, total_t = 0, 0
        # iterate over data
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # transfer to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # back-propogate loss
            optimizer.step() # update weight


            # calculate accuracy of predictions in the current batch
            _t, predicted_t = torch.max(outputs.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted_t == labels).sum().item()
            train_acc = 100. * correct_t/total_t

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('Epoch %d/%d | Mini-Batch %5d | Loss: %.3f | Accuracy: %.3f' % (epoch + 1, num_epochs, i + 1, running_loss / 2000, train_acc))
                losses.append(running_loss/2000)
                running_loss = 0.0
                #Accuracy
                accuracies.append(train_acc)

        lossesByEpoch.append(losses)
        trainAccByEpoch.append(accuracies)

        #######################################################################
        # 5. Test the network on the test data
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # We will check this by predicting the class label that the neural network
        # outputs, and checking it against the ground-truth. If the prediction is
        # correct, we add the sample to the list of correct predictions.

        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)        
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * float(correct / total)
        print('Accuracy of the network on the 10000 test images after epoch %d: %f %%' % (epoch + 1, accuracy))
        epoch_accuracy.append(accuracy)

    # plot losses
    plotLoss(lossesByEpoch, batchSize)

    # plot accuracy (100 - loss)
    plotAccuracy(trainAccByEpoch, batchSize)
    
    # plot accuracy
    plotBatchAccuracy(epoch_accuracy, batchSize)
    
    
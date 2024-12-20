import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4
seed  = 2025
epochs = 5
temperature = 3 # Softmax temperature
no_cuda = False
hard = False # Gumbel-softmax nature
is_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# classes = ("0","1","2","3","4","5","6","7","8","9")

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    """Sample code from CNN in pytorch.org"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # # MNIST
        # self.fc1 = nn.Linear(784, 250)
        # self.fc2 = nn.Linear(250, 100)
        # self.fc3 = nn.Linear(100, 10)
        
        # CIFAR10
        self.fc1 = nn.Linear(16 * 5* 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        # self.attention = nn.Parameter(torch.randn(latent_dim))
        

    def encode(self, x):
        # if not testing:
        #     x = gumbel_sampler(x, temperature)
        
        
        # CIFAR10
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        
        # h3 = gumbel_sampler(self.fc3(h2), temperature)
        h3 = self.fc3(h2)
        # if gumbel_is_apply:
        #     return nn.functional.gumbel_softmax(h3, temperature, hard)
        return h3


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        q = torch.flatten(x, 1) # flatten all dimensions except batch
        q = self.encode(q)

        #     q = F.gumbel_softmax(q, temperature, hard, dim=1)
        # print(q_y.size())

        return q

def evaluate_model(model, test_loader, device='cuda'):
    testing = True
    # sample code from pytorch.org
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    suma=0
    with torch.no_grad():
        running_loss = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            suma+=1

            # print statistics
            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    print(running_loss/suma)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    
    pass

def gumbel_sampler(input, tau):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise)
    x = (input + noise) / tau
    # x = F.softmax(x.view(-1, x.size(1)), dim=1)
    return x.view_as(input)


if __name__ == "__main__":
    

    
    print(f"Device: {device}")
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    latent_dim = len(classes)
    gumbel_is_apply = False
    batch = 2000
    
    net = Net()
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print("Now start training")
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # inputs = gumbel_sampler(inputs, temperature)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            

            ''' Comment out this line to perform original training'''
            outputs = gumbel_sampler(outputs,temperature)*0.1 + outputs
            
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % batch == batch-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        evaluate_model(net, testloader, device)
        temperature = temperature * 0.90
        PATH = './cifar_net_gumbel.pth'
    torch.save(net.state_dict(), PATH) 
    
    pass
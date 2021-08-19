# cifar100-Model
My first deep learning model, it's a classification model to classify a photo throw 100 class with the famous dataset cifar100.

# Requirements
Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

Python3 : https://www.python.org/downloads/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

GPU : Any GPU that is work with PyTorch 

# Model
* Hayper parametars
    
      starting learning rate = 0.01
    
      number of epochs = 200
    
      batch size = 1000

* Optimaizer 
    
      adam optimaizer

* Schedular 
    
      step size = 20

      gamma = 0.8

* Loss function
  
      cross entropy loss

* Model structure
    
      1. conv layer
      
      activation function (' ReLU ')
      
      max pooling
      
      batch norm
      
      drop out


      2. conv layer

      activation function (' ReLU ')

      max pooling
      
      batch norm
      
      drop out
    
    
      3. conv layer
      
      activation function (' ReLU ')
      
      max pooling
      
      batch norm
      
      drop out
      
      
      4. fully connected layer
    
      5. fully connected layer
    
      6. fully connected layer

# Accuracy 
    last epoch accuracy : 59.5
    
    best accuracy : 60.36

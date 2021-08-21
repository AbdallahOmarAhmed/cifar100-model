# cifar100-Model
My first deep learning model, it's a classification model to classify a photo throw 100 class with the famous dataset cifar100.

# Requirements
Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

Python3 : https://www.python.org/downloads/

numpy : https://numpy.org/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

GPU : Any GPU that is work with PyTorch 

DataSet : You should download the cifar100 dataset : https://www.cs.toronto.edu/~kriz/cifar.html

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

      1.conv layer ( input_channels = 3 , output_channels = 20 , kernal_size = 3)
      
      activation function (' ReLU ')
      
      max pooling ( kernal_size = 2 , stride = 2 )
      
      batch norm ( input_channels = 20 )
      
      drop out ( probability = 0.25 )


      2. conv layer ( 20 , 50 , 4 )

      activation function (' ReLU ')

      max pooling ( 2 , 2 )
      
      batch norm ( 50 )
      
      drop out ( 0.25 )
    
    
      3. conv layer ( 50 , 100 , 3 )
      
      activation function (' ReLU ')
      
      max pooling ( 2 , 2 )
      
      batch norm ( 100 )
      
      drop out ( 0.25 )
      
      
      4. fully connected layer ( input_features = 400 , output_features = 200 )
    
      5. fully connected layer ( 200 , 150 )
    
      6. fully connected layer ( 150 , 100 )

# Accuracy 
    last epoch accuracy : 60.75
    
    best accuracy : 61.26
    
  my model : https://drive.google.com/drive/folders/1LYjmtvBkNjfSTB71pFc1WCU2XL1uKwCv?usp=sharing
  
# Usage

* Read the data set 
    1. download the data set from this link : https://www.cs.toronto.edu/~kriz/cifar.html
    2. extract the data set file : https://stackoverflow.com/questions/48454111/how-to-extract-tar-files
    3. download the data_loader.py, in data_loader file you will find 2 classes ( cifar100DataSet , cifar100TestSet )
    5. in both of this classes you will find a variable called path, change it to match your data set path 

* Train the model 
    1. download the cifar100.py file
    2. " if you want ", you can change some parametars
    3. check that you have a GPU
    4. and now just run this file :)

* Load your model 
    1. first run the cifar100.py file
    2. you will find a file in the home directory called " model.pth "
    3. download the test_model.py file
    4. change the path variable to match the model.pth path
    5. choose how do you want to use this model


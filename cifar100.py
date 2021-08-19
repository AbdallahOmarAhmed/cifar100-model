import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_loader import cifar100DataSet, cifar100TestSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hayper parametars
num_epochs = 100
learning_rate = 0.001
batch_size = 1000

# data loader
train_data = cifar100DataSet()
test_data = cifar100TestSet()
# print(len(train_data))
# print(len(test_data))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

class cifar100Net(nn.Module):
    def __init__(self):
        super(cifar100Net, self).__init__()
        self.conv1 = nn.Conv2d(3,20,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(20,50,4)
        self.fc1 = nn.Linear(50*6*6,200)
        self.fc2 = nn.Linear(200,100)
        # self.layer2 = torch.nn.ModuleList(self.layer2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(-1,50*6*6)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = cifar100Net().to(device)
Loss = nn.CrossEntropyLoss()
optimaizer = torch.optim.Adam(model.parameters(), lr =learning_rate)

# train loop
for epoch in range(num_epochs):
    model.train()
    for x,y in train_dataloader:
        # forward
        images = x.to(device)
        labels = y.to(device)
        output = model(images)
        loss = Loss(output, labels)
        # backward
        optimaizer.zero_grad()
        loss.backward()
        optimaizer.step()

    print(epoch + 1, '/', num_epochs, 'loss = ', loss.item())
    model.eval()
    # accuracy
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        for x,y in test_dataloader:
            image = x.to(device)import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_loader import cifar100DataSet, cifar100TestSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hayper parametars
num_epochs = 100
learning_rate = 0.001
batch_size = 1000

# data loader
train_data = cifar100DataSet()
test_data = cifar100TestSet()
# print(len(train_data))
# print(len(test_data))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

class cifar100Net(nn.Module):
    def __init__(self):
        super(cifar100Net, self).__init__()
        self.conv1 = nn.Conv2d(3,20,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.norm1 = nn.BatchNorm2d(20)
        self.norm2 = nn.BatchNorm2d(50)
        self.norm3 = nn.BatchNorm2d(100)
        self.poolSame = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(20,50,4)
        self.conv3 = nn.Conv2d(50,100,3)
        self.fc1 = nn.Linear(100*2*2,200)
        self.fc2 = nn.Linear(200,150)
        self.fc3 = nn.Linear(150,100)
        self.dropout = nn.Dropout(0.25)
        # self.layer2 = torch.nn.ModuleList(self.layer2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm2(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm3(out)
        out = self.dropout(out)
        
        out = out.view(-1,100*2*2)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = cifar100Net().to(device)
Loss = nn.CrossEntropyLoss()
optimaizer = torch.optim.Adam(model.parameters(), lr =learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=20, gamma=0.8)
# train loop
for epoch in range(num_epochs):
    model.train()
    for x,y in train_dataloader:
        # forward
        images = x.to(device)
        labels = y.to(device)
        output = model(images)
        loss = Loss(output, labels)
        # backward
        optimaizer.zero_grad()
        loss.backward()
        optimaizer.step()
    scheduler.step()
    print(epoch + 1, '/', num_epochs, 'loss = ', loss.item())
    model.eval()
    # accuracy
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        for x,y in test_dataloader:
            image = x.to(device)
            label = y.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            n_total += 1
            n_correct += (pred == label).item()
        acc = 100.0 * n_correct / n_total
        print('accuracy = ',acc)



            label = y.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
def saveModle(model):
    
            n_total += 1
            n_correct += (pred == label).item()
        acc = 100.0 * n_correct / n_total
        print('accuracy = ',acc)
# save model
torch.save(model.state_dict(), 'model.pth')



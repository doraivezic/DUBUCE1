import torch
from torch import nn

import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

import skimage.io
import os
import math
import matplotlib.pylab as plt

 
#MODEL

class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, image_size, conv1_width, conv2_width, fc1_width, class_count):
    super(CovolutionalModel,self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool1=nn.MaxPool2d(kernel_size=2 ,stride=2)
    
    self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool2=nn.MaxPool2d(kernel_size=2 ,stride=2)
    
    fc_in = conv2_width * ( image_size/4. )**2   #/4 jer imamo 2 maxpool-a
    self.fc1 = nn.Linear( int(fc_in), fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool1(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    
    h = self.conv2(h)
    h = self.maxpool2(h)
    h = torch.relu(h)

    h = h.view(h.shape[0], -1)   #flatten
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    
    return logits


#CRTANJE

def draw_conv_filters(epoch, step, layer, save_dir):
  # C = layer.C
  # w = layer.weights.copy()
  # num_filters = w.shape[0]
  # k = int(np.sqrt(w.shape[1] / C))

  num_filters, C, k, k = layer[0].shape

  w = layer[0].data

  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % ('conv1', epoch, step, i)
    skimage.io.imsave(os.path.join(save_dir, filename), img)



#UCITAVANJE PODATAKA

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out_Convolutional_Model'

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
#train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))
train_y, valid_y, test_y = (y for y in (train_y, valid_y, test_y))

config = {}
#config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

train_data = []
for i in range(len(train_x)):
   train_data.append([train_x[i], train_y[i]])

valid_data = []
for i in range(len(valid_x)):
   valid_data.append([valid_x[i], valid_y[i]])

test_data = []
for i in range(len(test_x)):
   test_data.append([test_x[i], test_y[i]])

train_loader= DataLoader(dataset=train_data,batch_size=config['batch_size'])
validation_loader= DataLoader(dataset=valid_data,batch_size=config['batch_size'])
test_loader= DataLoader(dataset=test_data,batch_size=config['batch_size'])


#INICIJALIZACIJA ZA MODEL

inputs = np.random.randn(config['batch_size'], 1, 28, 28)

model = CovolutionalModel(1, 28, 16, 32, 512, 10)

criterion = nn.CrossEntropyLoss()

weight_decay = 1e-3   #regularizacija
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=weight_decay)



#TRAIN

n_epochs=10
cost_list=[]
accuracy_list=[]
N_test=len(valid_x)
cost=0
#n_epochs
for epoch in range(n_epochs):

    if epoch in config['lr_policy']:
      for g in optimizer.param_groups:
        g['lr'] = config['lr_policy'][epoch]['lr']

    cost=0    
    correct=0
    for i, (x, y) in enumerate(train_loader):

        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x)
        # calculate loss 
        loss=criterion(z,y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        cost+=loss.item()

        _, y_predicted = torch.max(z.data, 1)
        correct += (y_predicted == y).sum()

        if i % 5 == 0:
          print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*config['batch_size'], train_x.shape[0], loss.item()))
  
        if i > 0 and i % 50 == 0:
          print("Train accuracy = %.2f" % (correct / ((i+1)*config['batch_size']) * 100))
    
        if i % 100 == 0:
          draw_conv_filters(epoch, i*config['batch_size'], list(model.conv1.parameters()), config['save_dir'])
          
    print("Train accuracy = %.2f\n" % (correct / train_x.shape[0] * 100))
      
    cost_list.append(cost)



#VALIDATION

loss = 0.
correct = 0.

for i, (x, y) in enumerate(validation_loader):

    x = x.type(torch.FloatTensor)
    y = y.type(torch.LongTensor)

    z=model(x)
    loss+=criterion(z,y)

    _, y_predicted = torch.max(z.data, 1)
    correct += (y_predicted == y).sum()

print("Validation accuracy = %.2f" % (correct / valid_x.shape[0] * 100))
print("Avg loss = %.2f\n" % (loss / valid_x.shape[0]))
        
        

#TEST

loss = 0.
correct = 0.

for i, (x, y) in enumerate(test_loader):

    x = x.type(torch.FloatTensor)
    y = y.type(torch.LongTensor)

    z=model(x)
    loss+=criterion(z,y)

    _, y_predicted = torch.max(z.data, 1)
    #y_real = torch.max(y, 1)
    correct += (y_predicted == y).sum()

print("Test accuracy = %.2f" % (correct / test_x.shape[0] * 100))
print("Avg loss = %.2f\n" % (loss / test_x.shape[0]))



#CRTANJE KRETANJA GUBITKA KROZ EPOHE

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, 'Loss_epochs'))

plt.show()
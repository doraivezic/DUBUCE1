import os
import pickle
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn import metrics
import matplotlib.pyplot as plt

import skimage.io
import math


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

DATA_DIR = Path(__file__).parent / 'CIFAR'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)



#MODEL 
#conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)

class CIFAR_CovolutionalModel(nn.Module):
  def __init__(self, in_channels, image_size, conv1_width=16, conv2_width=32, fc1_width=256, fc2_width=128, class_count=10):
    super(CIFAR_CovolutionalModel,self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool1=nn.MaxPool2d(kernel_size=3 ,stride=2)
    
    self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool2=nn.MaxPool2d(kernel_size=3 ,stride=2)
    
    fc_in = conv2_width * ( (image_size-4)/4. )**2   #/4 jer imamo 2 maxpool-a
    self.fc1 = nn.Linear( int(fc_in), fc1_width, bias=True)
    self.fc2 = nn.Linear( fc1_width, fc2_width, bias=True)
    self.fc_logits = nn.Linear(fc2_width, class_count, bias=True)

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
    h = self.fc2(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    
    return logits



#Napišite funkciju evaluate koja na temelju predviđenih i točnih indeksa razreda određuje pokazatelje klasifikacijske performanse
def evaluate(y_predicted, y_real):

  confusion_matrix = metrics.confusion_matrix(y_real, y_predicted)

  TP = np.diag(confusion_matrix)

  FP = np.sum(confusion_matrix, axis=1) - TP

  FN = np.sum(confusion_matrix, axis=0) - TP

  TN = np.sum(confusion_matrix) - TP - FP - FN

  accuracy = (TP+TN) / (TP+TN+FP+FN)
  precision = TP / (TP+FP)
  recall = TP / (TP+FN)

  return accuracy, confusion_matrix, precision, recall




#DRAWING THE WEIGHTS MATRIX

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  skimage.io.imsave(os.path.join(save_dir, filename), img)





#UCITAVANJE PODATAKA

batch_size = 50

train_data = []
for i in range(len(train_x)):
   train_data.append([train_x[i], train_y[i]])

valid_data = []
for i in range(len(valid_x)):
   valid_data.append([valid_x[i], valid_y[i]])

test_data = []
for i in range(len(test_x)):
   test_data.append([test_x[i], test_y[i]])

train_loader =      DataLoader( dataset=train_data, batch_size=batch_size)
validation_loader = DataLoader( dataset=valid_data, batch_size=batch_size)
test_loader =       DataLoader( dataset=test_data,  batch_size=batch_size)



#INICIJALIZACIJA ZA MODEL

SAVE_DIR = Path(__file__).parent / 'out_CIFAR'

inputs = np.random.randn( batch_size, num_channels, img_height, img_width )

model = CIFAR_CovolutionalModel(num_channels, img_height)
draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)

criterion = nn.CrossEntropyLoss()

weight_decay = 1e-3   #regularizacija
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=weight_decay)
#za padajucu stopu ucenja mozemo koristiti torch.optim.lr_scheduler.ExponentialLR
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)



##TRAIN+VALIDATION

n_epochs=30
cost_list_train=[]
cost_list_validation=[]
accuracy_list_train=[]
accuracy_list_validation=[]
N_test=len(valid_x)
cost=0
lr_list = []

y_pred = []
y_real = []

for epoch in range(n_epochs):

    print("\nEpoch %d, learning rate %.2f\n" %(epoch, scheduler.get_last_lr()[0]) )
    lr_list.append(scheduler.get_last_lr())


    #TRAIN

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

        y_real.extend(y)
        y_pred.extend(y_predicted)
    
        # if i % 100 == 0:
        #   draw_conv_filters(epoch, i*config['batch_size'], list(model.conv1.parameters()), config['save_dir'])

    print("Train accuracy = %.2f" % (correct / train_x.shape[0] * 100))
    print("Avg loss = %.2f" % (loss / train_x.shape[0]))
      
    cost_list_train.append(cost)
    accuracy_list_train.append(correct / train_x.shape[0])

    accuracy, confusion_matrix, precision, recall = evaluate(y_pred, y_real)
    print("Precision %.2f, Recall %.2f\n" % (np.mean(precision), np.mean(recall)))



    #VALIDATION

    loss = 0.
    correct = 0.

    y_pred.clear()
    y_real.clear()

    for i, (x, y) in enumerate(validation_loader):

        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        z=model(x)
        loss+=criterion(z,y)
        cost+=loss.item()

        _, y_predicted = torch.max(z.data, 1)
        correct += (y_predicted == y).sum()

        y_real.extend(y)
        y_pred.extend(y_predicted)

    print("Validation accuracy = %.2f" % (correct / valid_x.shape[0] * 100))
    print("Avg loss = %.2f" % (loss / valid_x.shape[0]))

    cost_list_validation.append(cost)
    accuracy_list_validation.append(correct / valid_x.shape[0])

    scheduler.step()

    accuracy, confusion_matrix, precision, recall = evaluate(y_pred, y_real)
    print("Precision %.2f, Recall %.2f\n" % (np.mean(precision), np.mean(recall)))


draw_conv_filters(n_epochs, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)



fig, axs = plt.subplots(2, 2)

# Plot train and validation loss on the first subplot
axs[0, 0].plot(cost_list_train, '-o', label='Train Loss')
axs[0, 0].plot(cost_list_validation, '-o', label='Validation Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# Plot train and validation accuracy on the second subplot
axs[0, 1].plot(accuracy_list_train, '-o', label='Train Acc')
axs[0, 1].plot(accuracy_list_validation, '-o', label='Validation Acc')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

# Plot learning rate on the third subplot
axs[1, 0].plot(lr_list, '-o', label='Learning Rate')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Learning Rate')
axs[1, 0].legend()

# Fourth subplot is left empty

# Adjust layout and show the plots
fig.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Graphs'))
# plt.show()




#Prikažite 20 netočno klasificiranih slika s najvećim gubitkom te ispišite njihov točan razred, 
# kao i 3 razreda za koje je model dao najveću vjerojatnost

# Da biste prikazali sliku, morate najprije poništiti normalizaciju srednje vrijednosti i varijance:
import skimage as ski
import skimage.io

def draw_image(img, mean, std):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()


misclassified_images = []
misclassified_losses = []
misclassified_outputs = []
missclassified_real_output_class = []

criterion = nn.CrossEntropyLoss(reduction='none')

with torch.no_grad():

  for i, (x, y) in enumerate(validation_loader):

        x = x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)

        z=model(x)
        loss=criterion(z,y)

        _, y_predicted = torch.max(z.data, 1)
        not_correct = torch.nonzero(y_predicted != y, as_tuple=False)

        for i in not_correct:
            misclassified_images.append(x[i])
            misclassified_losses.append(loss[i])
            misclassified_outputs.append(z[i])
            missclassified_real_output_class.append(y[i])

misclassified_images = torch.stack(misclassified_images).numpy()
misclassified_losses = np.array(misclassified_losses)
missclassified_real_output_class = torch.stack(missclassified_real_output_class).numpy().reshape(-1)
misclassified_outputs = torch.stack(misclassified_outputs).numpy()

max_loss_indices = np.argsort(misclassified_losses)[-20:]

label_names = unpickle(os.path.join(DATA_DIR, 'batches.meta'))['label_names']

for el in max_loss_indices:

  max_outputs = np.argsort(misclassified_outputs[int(el)].reshape(-1))[-3:]
  
  labels = ""
  for output in max_outputs[::-1]:  #od najvise vjerojatne klase do najmanje
    labels += "_" + label_names[output]

  correct_label_index = missclassified_real_output_class[int(el)].reshape(-1)[0]
  correct_label = label_names[correct_label_index]
  filename = "Image_%d_Outputs_corr_%s_predicted%s.png" % (el, correct_label, labels )

  img = misclassified_images[el].reshape(num_channels,img_height,img_width)

  img = img.transpose(1, 2, 0)
  img *= data_std
  img += data_mean
  img = img.astype(np.uint8)
  #skimage.io.imshow(img)
  #skimage.io.show()

  skimage.io.imsave(os.path.join(SAVE_DIR, filename), img)
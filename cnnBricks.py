# from https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/

# importing the libraries
import pandas as pd
import numpy as np
import math

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# loading dataset
train = pd.read_csv('training-balanced.csv')
test = pd.read_csv('testing.csv')

sample_submission = pd.read_csv('development.csv')

train.head()

# loading training images
train_img = []
for img_name in tqdm(train['File']):
    # defining the image path
    image_path = 'Images/' + str(img_name)
    # reading the image
    img = imread(image_path, as_gray=True)
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
# defining the target
train_y = train['Labels'].values
train_x.shape

# # visualizing images
# i = 0
# plt.figure(figsize=(10, 10))
# plt.subplot(221), plt.imshow(train_x[i], cmap='gray')
# plt.subplot(222), plt.imshow(train_x[i + 25], cmap='gray')
# plt.subplot(223), plt.imshow(train_x[i + 50], cmap='gray')
# plt.subplot(224), plt.imshow(train_x[i + 75], cmap='gray')

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

# converting training images into torch format
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
train_x = torch.from_numpy(train_x)

# converting the target into torch format
train_y = torch.LongTensor(train_y.astype(int))

# shape of training data
# print(train_x.shape, train_y.shape)

# converting validation images into torch format
val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])
val_x = torch.from_numpy(val_x)

# converting the target into torch format
val_y = torch.LongTensor(val_y.astype(int))

# shape of validation data
# print(val_x.shape, val_y.shape)

# downsample the dataset
train_x = train_x[:20]
train_y = train_y[:20]


# Let’s define the architecture:
class Net(Module):
    def __init__(self, num_features=4):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            # Linear(4 * 7 * 7, 10) # input_num_units, hidden_num_units
            Linear(num_features * 128 * 128, num_features)  # 128 is H_out from MaxPool2d twice
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print('model stored on %s' % str(next(model.parameters()).device))


# This is the architecture of the model. We have two Conv2d layers and a Linear layer.

def mem():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 20
    return str(memoryUse)


# Next, we will define a function to train the model:
def train(epoch):
    print(mem())
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    # train_losses.append(loss_train) # memory issues
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)


# split workload across all workers
class MyIterableDataSet(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataSet).__init__()
        assert end > start, "works only with end>=start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

ds = MyIterableDataSet(start=3, end=7)

# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    # for img
    train(epoch)

# plotting the training and validation loss
plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()


def make_prediction(input_for_pred):
    with torch.no_grad():
        output = model(input_for_pred)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    return predictions


# prediction for training set
predictions = make_prediction(train_x.cuda())
# accuracy on training set
accuracy_score(train_y, predictions)

# prediction for validation set
predictions = make_prediction(val_x.cuda())
# accuracy on validation set
accuracy_score(val_y, predictions)

# loading test images
test_img = []
for img_name in tqdm(test['File']):
    # defining the image path
    image_path = 'Images/' + str(img_name)
    # reading the images
    img = imread(image_path, as_gray=True)
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)

# converting the list to numpy array
test_x = np.array(test_img)
test_x.shape

# converting training images into torch format
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])
test_x = torch.from_numpy(test_x)
test_x.shape

# generating predictions for test set
predictions = make_prediction(test_x.cuda())

# TODO : There is something wrong with development.csv (too long for pred)
# # replacing the label with prediction
# sample_submission['Label'] = predictions
# sample_submission.head()
#
# # saving the file
# sample_submission.to_csv('development.csv', index=False)

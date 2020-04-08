"""ArchB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qQFDjf7oss4umHCTUKdwKYLEj28a5SUA
"""

# Importing the support for the model
import numpy as np
import os
import tensorflow as tf
import keras


# print(os.environ)
os.environ["KERAS_BACKEND"] = 'tf'

# Importing model and image pre-processing
from keras.models import Sequential
import keras.backend as K

K.set_image_data_format("channels_last")
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Importing Core layer
from keras.layers.core import Dense, Dropout, Activation, Flatten

# Importing Convolution layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# Importing Preprocessing platforms
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

# Importing plotting
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# import PIL

# Inputting image dimenstions
img_rows, img_cols = 150, 150
img_channels = 3  # rgb
epochs = 400
batch_size = 20

# Data was already split into train test and were saved in respective folders
# Read the train test folders
# Labels train and test
# Locate the csv files of train and test
train_csv = pd.read_csv("training.csv")
test_csv = pd.read_csv(f"testing.csv")

# As the data generator takes the categorical texts,changed earlier 0 and 1 to cracked and uncracked
train_csv['label'] = ["cracked" if x == 1 else "uncracked" for x in train_csv.Labels]
test_csv['label'] = ["cracked" if x == 1 else "uncracked" for x in test_csv.Labels]

data_path = "Images"
# train_path= "./Data/training"
# test_path = "./Data/test"
print(train_csv)

# TRAIN (Generating Image Data Generator class arguments)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    validation_split=0.25)

# TEST (Generating Image Data Generator class arguments)
test_datagen = ImageDataGenerator(rescale=1. / 255, data_format="channels_last")

# TRAIN and TEST data generator ; takes datafame
# and the parth to a directory and generates batches
# of augumented data

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory=data_path,
    color_mode='rgb',
    x_col='File',
    y_col="label",
    subset="training",
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="categorical"
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_csv,
    directory=data_path,
    color_mode='rgb',
    x_col='File',
    y_col="label",
    subset="validation",
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    directory=data_path,
    color_mode='rgb',
    x_col='File',
    y_col="label",
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode="categorical"
)

if K.image_data_format() == 'channels_first':
    input_shape = (img_channels, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, img_channels)

print(train_generator.n)

# from IPython.display import clear_output


class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        # clear_output(wait=True)

        # ax1.set_yscale('log')
        ax1.plot(self.x, self.accuracy, label="acc")
        ax1.plot(self.x, self.val_accuracy, label="val_acc")
        ax1.legend()

        ax2.plot(self.x, self.losses, label="loss")
        ax2.plot(self.x, self.val_losses, label="val_loss")
        ax2.legend()

        plt.show();


plot = PlotLearning()

# Develop models and start training

# From paper the sequence of Architecture B (Hybrid architecture):
# 1.  2D conv of size 15x15, filter depth= 3, number= 20, padding = 'same' stride=1
# 2.  ReLU Activation
# 3.  2D conv of size 10x10, depth=20, num = 20, padding ='Valid', stride =1
# 4.  ReLU Activation
# 5.  Max Pool of size 5x5, depth= 20, stride= 5
# 6.  Dropout--0.25
# 7.  2D conv of size 5x5, depth=20, num = 20, padding ='Same', stride =1
# 8.  ReLU Activation
# 9.  2D conv of size 2x2, depth=40, num = 40, padding ='Valid', stride =1
# 10. ReLU Activation
# 11. Max Pool of size 4x4, depth= 40, stride= 4
# 12. Dropout--0.25
# 13. Flatten
# 14. Dense Number of filters = 50
# 15. ReLU Activation
# 16. Dropout--0.5
# 17. Dense Number of filters = 2
# 18.Softmax

# Define parameters for Architecture B
strides1 = (5, 5)
strides2 = (4, 4)

model = Sequential()
model.add(Convolution2D(20, 15, strides=(1, 1),
                        padding='same',
                        input_shape=input_shape, activation='relu'))
model.add(Convolution2D(20, 10, strides=(1, 1),
                        padding='valid',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=strides1))
model.add(Dropout(0.25))
model.add(Convolution2D(40, 5, strides=(1, 1), padding="same", activation='relu'))
model.add(Convolution2D(40, 2, strides=(1, 1), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=strides2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(RMSprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
mymodel = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=validation_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=100,
                              # callbacks=[plot]

                              )

# from keras.callbacks import callacks
# mymodel= callbacks.History
train_loss = mymodel.history['loss']
val_loss = mymodel.history['val_loss']
train_acc = mymodel.history['accuracy']
val_acc = mymodel.history['val_accuracy']
xc = range(100)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs Val_loss')
plt.grid(True)
plt.legend(['train', 'test'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs Val_acc')
plt.grid(True)
plt.legend(['train', 'test'])
plt.style.use(['classic'])
plt.show()

# model.save(ArchB_model.h5)
model.summary()

model.save_weights("ArchB_model.h5")

model.save("archi_B", include_optimizer=True)

score = model.evaluate(test_generator)
print(score[0], 'Test loss')
print(score[1], 'Test Accuracy')

model_json = model.to_json()
with open("./ArchB_model.json", "w") as json_file:
    json_file.write(model_json)

from keras.models import model_from_json

# opening and storing file in a variable
json_file = open('./ArchB_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

mymodel = model_from_json(loaded_model_json)

# load weights into new model
mymodel.load_weights("./ArchB_model.h5")
print("Loaded Model from folder")

# compile and evaluate loaded model

mymodel = model.compile(RMSprop(lr=0.001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

from keras.callbacks import CSVLogger

csv_logger = CSVLogger("model_history_log.csv", append=True)
mymodel = model.fit_generator(train_generator, callbacks=[csv_logger])




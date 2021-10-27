from __future__ import division
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
# from keras.util import plot_model
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

print( 'Using Keras version', keras.__version__)

output_dir = "output_30"
batch_size = 32

#data resolution
img_rows, img_cols, channels = 256, 256, 3
input_shape = (img_rows, img_cols, channels)

gendata_train = ImageDataGenerator(rescale=1./255)

traindata = gendata_train.flow_from_directory(directory="Data/Train", target_size=(img_rows, img_cols), batch_size=batch_size, shuffle=True, color_mode='rgb')

gendata_valid = ImageDataGenerator(rescale=1./255)
validdata = gendata_valid.flow_from_directory(directory="Data/Valid", target_size=(img_rows, img_cols), batch_size=batch_size, shuffle=False, color_mode='rgb')

print(traindata.class_indices)
print(validdata.class_indices)
#Define the NN architecture
model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=256,activation="relu", kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
model.add(Dropout(0.3))
model.add(Dense(units=256,activation="relu", kernel_regularizer =tf.keras.regularizers.l2( l=0.01)))
model.add(Dense(units=29, activation="sigmoid"))

#Compile the NN
opt = Adam(learning_rate=0.000001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# model.load_weights('vgg16_0050_0.5639.h5')

#Start training
checkpoint = ModelCheckpoint('vgg30.h5', monitor='val_accuracy',
                             verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1, mode='auto')

history = model.fit(
    traindata,
    steps_per_epoch=traindata.samples // batch_size,
    validation_data=validdata,
    validation_steps=validdata.samples // batch_size,
    epochs=60,
    callbacks=[checkpoint, early])

#Evaluate the model with test set
# score = model.evaluate(x_test, y_test, verbose=0)
# print('test loss:', score[0])
# print('test accuracy:', score[1])

##Store Plots

#Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(f'{output_dir}/vgg_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(f'{output_dir}/vgg_loss.pdf')

#Confusion Matrix

# #Compute probabilities
Y_pred = model.predict_generator(validdata, steps = len(validdata.filenames))
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred[0])
print(y_pred)
print(len(y_pred))
print(validdata.classes)
print(len(validdata.classes))
#Plot statistics
print( 'Analysis of results' )
cr_matrix = classification_report(validdata.classes, y_pred)
print(cr_matrix)
cf_matrix = confusion_matrix(validdata.classes, y_pred)
print(cf_matrix)

plt.clf()
sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g')
plt.savefig(f"{output_dir}/confusion_matrix.png")

#
# #Saving model and weights
# from keras.models import model_from_json
# model_json = model.to_json()
# with open(f'{output_dir}/model.json', 'w') as json_file:
#         json_file.write(model_json)
# weights_file = f"{output_dir}/weights-MNIST_"+str(score[1])+".hdf5"
# model.save_weights(weights_file,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)


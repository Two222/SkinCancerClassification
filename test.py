from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras import callbacks
from keras.models import load_model

import matplotlib.pyplot as plt


train_data_dir = 'data/train_data2'
val_dir = 'data/val_data2'
nb_train_samples = 2508
nb_test = 150
epochs = 50
img_width, img_height = 224, 224


model = load_model('weights.03-0.49.hdf5')

test_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = test_datagen.flow_from_directory(
                    val_dir,
                    target_size=(img_width, img_height),
                    batch_size=30,
                    class_mode='binary',
                    shuffle = False)

test_labels = np.array([0]*108 + [1]*42)
preds = model.predict_generator(val_generator,nb_test//30)

preds[preds > 0.5] = 1
preds[preds <= 0.5] = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(preds)):
	if preds[i] == test_labels[i] and preds[i] == 1:
		tp += 1
	elif preds[i] == test_labels[i] and preds[i] == 0:
		tn += 1
	elif preds[i] != test_labels[i] and preds[i] == 1:
		fp += 1
	elif preds[i] != test_labels[i] and preds[i] == 0:
		fn += 1
print(preds[1:20])
print(test_labels[1:20])
print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))
print("Precision:",(tp/(tp+fp)))
print("Recall:",(tp/(tp+fn)))
print(tp,fp,tn,fn)		


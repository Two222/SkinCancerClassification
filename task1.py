
# coding: utf-8

# In[3]:

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

import matplotlib.pyplot as plt

np.random.seed(10)

# In[12]:

img_size = 48
nb_train = 1626+1496
nb_test = 150
train_dir = 'data/train_data1'
val_dir = 'data/val_data1'
top_model_weights_path = 'bottleneck_fc_model.h5'


epochs = 50
batch_size = 32
input_shape = (img_size,img_size,3)


class Batch_Loss(callbacks.Callback):

    def on_train_begin(self,logs = {}):
        self.losses = []

    def on_batch_end(self,batch,logs = {}):
        self.losses.append(logs.get('loss'))


# In[9]:

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = input_shape,padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),input_shape = input_shape,padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),input_shape = input_shape,padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),input_shape = input_shape,padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])


# In[14]:
batch_loss = Batch_Loss()
best_checkpoints = callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_acc',verbose = 1,save_best_only = True)

train_datagen = ImageDataGenerator(
                rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size = (img_size,img_size),
                batch_size = batch_size,
                class_mode='binary')

val_generator = test_datagen.flow_from_directory(
                    val_dir,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='binary')


# In[ ]:

history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=nb_test // batch_size,
            callbacks = [batch_loss,best_checkpoints])

model.save_weights(top_model_weights_path)

plt.plot(batch_loss.losses)
plt.show()

plt.plot(history.history['val_loss'])
plt.show()

'''
preds = model.predict(test_data)
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
print("Accuracy:",(tp+tn)/(tp+tn+fp+fn))
print("Precision:",(tp/(tp+fp)))
print("Recall:",(tp/(tp+fn)))
print(tp,fp,tn,fn)      

'''

# In[ ]:




from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, adagrad
from keras.utils import plot_model
import keras
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split

# data generator
"""
data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

vali_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

train_gen = data_gen.flow(X_train, y_train, batch_size=16)
vali_gen = vali_gen.flow(X_test, y_test, batch_size=16)
"""

train_gen_set = ImageDataGenerator(rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)


test_gen_set = ImageDataGenerator(rescale=1./255)

train_dir = r'D:\dogs-vs-cats_small\train'
vali_dir = r'D:\dogs-vs-cats_small\validation'

train_gen = train_gen_set.flow_from_directory(train_dir, 
                target_size=(150, 150), 
                class_mode='binary', 
                batch_size=20)

vali_gen = test_gen_set.flow_from_directory(vali_dir, 
                target_size=(150, 150), 
                class_mode='binary', 
                batch_size=20)

# inception v3
base_model = InceptionV3(weights='imagenet', include_top=False)

#base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs=prediction)

model.summary()

def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
#    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

#setup_to_transfer_learning(model, base_model)
setup_to_fine_tune(model, base_model)



"""
# Tensorboard
callbacks = [keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
            )]
"""

history = model.fit_generator(train_gen,
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=vali_gen,
                    validation_steps=15,
#                    callbacks=callbacks
                    )


"""
model.save('ft_inception.h5')

import numpy as np
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
"""


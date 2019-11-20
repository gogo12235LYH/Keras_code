
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, BatchNormalization, concatenate, Input
from keras.utils import to_categorical, plot_model
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

#base = InceptionV3(weights='imagenet', include_top=False)

bb1 = VGG16(weights='imagenet')
bb1.summary()

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    featurewise_center=True
)

vali_gen = ImageDataGenerator(
    rescale=1./255
)


x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

train_gen = data_gen.flow(X_train, y_train, batch_size=32)
vali_gen = vali_gen.flow(X_test, y_test, batch_size=32)
X_test = X_test/255.

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# define conv_base_bn
def conv_bn(x, filter_nb, k_size, strides=(1,1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None

    x = Conv2D(filter_nb, k_size, strides=strides, padding=padding, activation = 'relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)

    return x

def Inception_block(x, previous_layer_parameter):

    (branch1, branch2, branch3, branch4) = previous_layer_parameter

    # branch 1
    branch1_1x1 = Conv2D(branch1[0], (1,1), strides=(1,1), padding='same', activation='relu')(x)

    # branch 2
    branch2_1x1 = Conv2D(branch2[0], (1,1), strides=(1,1), padding='same', activation='relu')(x)
    branch2_3x3 = Conv2D(branch2[1], (3,3), strides=(1,1), padding='same', activation='relu')(branch2_1x1)

    # branch 3
    branch3_1x1 = Conv2D(branch3[0], (1,1), strides=(1,1), padding='same', activation='relu')(x)
    branch3_3x3 = Conv2D(branch3[1], (3,3), strides=(1,1), padding='same', activation='relu')(branch3_1x1)

    # branch 4
    branch4_pooling = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    branch4_1x1 = Conv2D(branch4[0], (1,1), strides=(1,1), padding='same', activation='relu')(branch4_pooling)
    
    # connect 4 branchs to output, each layer's shape are (None, x, x, x)
    x = concatenate([branch1_1x1, branch2_3x3, branch3_3x3, branch4_1x1], axis=3)

    return x

def Inception_V1(input_shape, label_shape):

    input_tensor = Input(shape=input_shape)

    x = conv_bn(input_tensor, 64, (7,7), strides=(2,2), padding='same')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = conv_bn(x, 192, (3,3), strides=(1,1))
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    x = Inception_block(x,[(64,),(96,128),(16,32),(32,)])               #Inception 3a 28x28x256
    x = Inception_block(x,[(128,),(128,192),(32,96),(64,)])             #Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)   #14x14x480

    x = Inception_block(x,[(192,),(96,208),(16,48),(64,)])              #Inception 4a 14x14x512
    x = Inception_block(x,[(160,),(112,224),(24,64),(64,)])             #Inception 4a 14x14x512
    x = Inception_block(x,[(128,),(128,256),(24,64),(64,)])             #Inception 4a 14x14x512
    x = Inception_block(x,[(112,),(144,288),(32,64),(64,)])             #Inception 4a 14x14x528
    x = Inception_block(x,[(256,),(160,320),(32,128),(128,)])           #Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)   #7x7x832

    x = Inception_block(x,[(256,),(160,320),(32,128),(128,)])           #Inception 5a 7x7x832
    x = Inception_block(x,[(384,),(192,384),(48,128),(128,)])           #Inception 5b 7x7x1024

    x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(label_shape, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model



IMG_SIZE = X_train.shape[1]
IMG_C = X_train.shape[3]
CLASS_NUM = y_train.shape[1]

model = Inception_V1((IMG_SIZE, IMG_SIZE, IMG_C),(CLASS_NUM))
model.summary()

model.compile(optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                loss = 'categorical_crossentropy',
                metrics=['acc'])

#plot_model(model, to_file='Inception.png')

"""
history = model.fit_generator(train_gen, steps_per_epoch = 100, validation_data=(X_test, y_test), epochs = 25, validation_steps=15)


model.save('inceptionv1_flowers.h5')

import numpy as np
import matplotlib.pyplot as plt

#epochs = np.range(100)
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
















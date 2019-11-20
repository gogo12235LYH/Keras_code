import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

epochs = 200
NB = 56
LR = 1e-5
BN = 16

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True)

train_data = X_train[200:]
train_label = y_train[200:]

vali_data = X_train[:200]
vali_label = y_train[:200]

"""
------------- data generator -----------
"""
train_generator = ImageDataGenerator(
#                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
                )

vali_gen = ImageDataGenerator(
#                           rescale=1./255
                            )

train_gen = train_generator.flow(train_data, train_label, batch_size=BN)

vali_gen_data = vali_gen.flow(vali_data, vali_label, batch_size=BN)

test_gen = vali_gen.flow(X_test, y_test, batch_size=BN)


base_ = ResNet50(weights='imagenet', pooling='avg', include_top=False)

input_tensor = base_.input
out = base_.output
out = Dropout(0.5)(out)
out = Dense(17, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001))(out)
model = Model(inputs=input_tensor, outputs=out)

print(len(base_.trainable_weights))
base_.trainable = False
print(len(base_.trainable_weights))

"""
set_ = False
for layer in model.layers:
    if layer.name == 'conv1_pad':
        set_ = False
    if set_ is not True:
        layer.trainable = False
    if layer.name == 'global_average_pooling2d_1':
        break
"""
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=LR),
              metrics=['acc']
              )

model.summary()

lr_data = []

def lr_scheduler(epoch):
    if epoch >= 50:
        lr = LR / 4
    elif epoch >= 25:
        lr = LR / 2
    else:
        lr = LR
    lr_data.append(lr)
    return lr

scheduler = LearningRateScheduler(lr_scheduler)

callbacks = [scheduler, 
#            checkpoint
            ]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_data)/BN,
                    epochs=epochs,
                    validation_data=vali_gen_data,
                    validation_steps=len(vali_data)/BN,
                    callbacks=callbacks
                    )


loss_e, acc_e = model.evaluate_generator(test_gen)

print('Loss : ',loss_e, ' acc : ', acc_e)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']


plt.plot(acc)
plt.plot(val_acc, 'go')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\resnet50\resnet_acc_{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'go')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\resnet50\resnet_loss_{}.png'.format(NB))
plt.show()

model.save('resnet_flower_{}.h5'.format(NB))










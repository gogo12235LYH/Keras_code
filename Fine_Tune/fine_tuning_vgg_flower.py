from keras.models import Model, load_model
from keras.layers import Flatten, Activation, GlobalAveragePooling2D, Dense, Dropout
from keras.applications.vgg16 import VGG16
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Ear
import keras
from keras.metrics import top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

epochs = 50
NB = 3722

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True)

#train_data = X_train[200:]
#train_label = y_train[200:]

#vali_data = X_train[:200]
#vali_label = y_train[:200]


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


train_gen = train_generator.flow(train_data, train_label, batch_size=20)

vali_gen_data = vali_gen.flow(vali_data, vali_label, batch_size=20)

test_gen = vali_gen.flow(X_test, y_test, batch_size=20)

model = load_model('D:\\code\\deep_learning\\Keras_practice\\Fin_vgg16_flower_3721.h5',
                   custom_objects = { 'top_5_accuracy': top_5_accuracy, 'top_1_accuracy': top_1_accuracy}
                   )

#print(len(base_.trainable_weights))
#base_.trainable = False
#model.trainable = True

set_ = False
for layer in model.layers:
    if layer.name == 'block3_conv1':
        set_ = True
    if set_ :
        layer.trainable = True
    else:
        layer.trainable = False
        
#print(len(base_.trainable_weights))

LR_ = 1e-5

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LR_), metrics=['acc',
                                                                                 top_1_accuracy,
                                                                                 top_5_accuracy
                                                                                 ])

lr_data = []


def lr_scheduler(epoch):
    if epoch > 0.9 * epochs:
        lr = LR_ / 8
    elif epoch > 0.75 * epochs:
        lr = LR_ / 4
    elif epoch > 0.5 * epochs:
        lr = LR_ / 2
    elif epoch > 0.25 * epochs:
        lr = LR_ / 2
    else:
        lr = LR_
    lr_data.append(lr)
    return lr

scheduler = LearningRateScheduler(lr_scheduler)

callbacks = [scheduler, 
#            checkpoint
            ]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_data)/20,
                    epochs=epochs,
                    validation_data=vali_gen_data,
                    validation_steps=len(vali_data)/20,
                    callbacks=callbacks
                    )

loss_e, acc_e , top_1, top_5 = model.evaluate_generator(test_gen)

print('Loss : ', loss_e, ' acc : ', acc_e, ' Top 1 acc : ', top_1, ' Top 5 acc : ', top_5)

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
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\acc_{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'go')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\loss_{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['top_5_accuracy'])
plt.plot(history.history['val_top_5_accuracy'], 'go')
plt.title('model learning rate')
plt.ylabel('Top_5_categorical_accuracy')
plt.xlabel('epoch')
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\top_5_acc{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['top_1_accuracy'])
plt.plot(history.history['val_top_1_accuracy'], 'go')
plt.title('model learning rate')
plt.ylabel('Top_1_categorical_accuracy')
plt.xlabel('epoch')
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\models\flower_plot\top_1_acc{}.png'.format(NB))
plt.show()

model.save('Fin_vgg16_flower_{}.h5'.format(NB))


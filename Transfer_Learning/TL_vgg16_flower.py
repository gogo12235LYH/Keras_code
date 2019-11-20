from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.vgg16 import VGG16
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras


epochs = 30
NB = 37

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True)

train_data = X_train[200:]
train_label = y_train[200:]

vali_data = X_train[:200]
vali_label = y_train[:200]


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

base_ = VGG16(weights='imagenet', include_top=False, pooling='avg')

input_tensor = base_.input

out = base_.output
out = Dropout(0.5)(out)
#out = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(out)
#out = Dropout(0.35)(out)
out = Dense(17, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001))(out)

model = Model(inputs=input_tensor, outputs=out)


print(len(base_.trainable_weights))
base_.trainable = False
"""
base_.trainable = True
set_ = False
for layer in base_.layers:
    if layer.name == 'block5_conv1':
        set_ = True
    if set_ :
        layer.trainable = True
    else:
        layer.trainable = False
"""
print(len(base_.trainable_weights))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-5), metrics=['acc'])


lr_data = []

def lr_scheduler(epoch):
    if epoch > 0.75 * epochs:
        lr = 1e-5 / 4
    elif epoch > 0.5 * epochs:
        lr = 1e-5 / 2
    else:
        lr = 1e-5
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


model.save('vgg_flower_{}.h5'.format(NB))
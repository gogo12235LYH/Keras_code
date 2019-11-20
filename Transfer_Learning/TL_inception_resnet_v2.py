
from keras.models import Model, Input
from keras.layers import Flatten, Activation, GlobalAveragePooling2D, Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
import keras

def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

epochs = 50
lr_data = []

def lr_scheduler(epoch):
    if epoch > 0.9 * epochs:
        lr = 0.00001 / 8
    elif epoch > 0.75 * epochs:
        lr = 0.00001 / 4
    elif epoch > 0.5 * epochs:
        lr = 0.00001 / 2
    elif epoch > 0.25 * epochs:
        lr = 0.00001 / 2
    
    else:
        lr = 1e-4
    lr_data.append(lr)
    return lr



scheduler = LearningRateScheduler(lr_scheduler)

data_gen = ImageDataGenerator(
#    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

vali_gen = ImageDataGenerator(
#   rescale=1./255,
)

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

train_data = X_train[200:]
train_label = y_train[200:]

vali_data = X_train[:200]
#vali_data = vali_data /255.

vali_label = y_train[:200]
#vali_label = vali_label / 255.


train_gen = data_gen.flow(train_data, train_label, batch_size=16)

vali_gen_data = vali_gen.flow(vali_data, vali_label, batch_size=16)

test_gen = vali_gen.flow(X_test, y_test, batch_size=16)

base_model = InceptionResNetV2( weights='imagenet',
                                 include_top=False,
                                 pooling='avg'
#                                 input_shape= (150, 150, 3)
                                 )

out = base_model.output
out = Dropout(0.5)(out)
out = Dense(500, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(out)
out = Dropout(0.5)(out)
out = Dense(256, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(17, activation='softmax')(out)
model = Model(inputs=base_model.input, outputs=out)

#model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy',metrics=['accuracy'])

print(len(base_model.trainable_weights))
base_model.trainable = False
"""
GG = 780
for layer in base_model.layers[:GG+1]:
    layer.trainable = False

for layer in base_model.layers[GG+1:]:
    layer.trainable = True
"""
print(len(base_model.trainable_weights))

model.compile(optimizer=RMSprop(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy', 
              top_k_accuracy]
              )

checkpoint = ModelCheckpoint("./ckpts1/1232weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5",
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

callbacks = [scheduler, 
#            checkpoint
            ]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_data)/16,
                    epochs=epochs,
                    validation_data=vali_gen_data,
                    validation_steps=len(vali_data)/16,
#                    callbacks=callbacks
                    )

loss_e, acc_e, top_e = model.evaluate_generator(test_gen)

print('Loss : ',loss_e, ' acc : ', acc_e, ' Top 5 acc : ', top_e)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

#model.save('inception_resnet_v2__1.h5')

NB = 41

plt.plot(acc)
plt.plot(val_acc, 'go')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch (68 steps)')
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\plot_results\inception_resnet_v2_transfer_learning\acc_{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'go')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch (68 steps)')
plt.legend(['train', 'validation'])
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\plot_results\inception_resnet_v2_transfer_learning\loss_{}.png'.format(NB))
plt.show()

plt.clf()
plt.plot(history.history['top_k_categorical_accuracy'])
plt.plot(history.history['val_top_k_categorical_accuracy'], 'go')
plt.title('model learning rate')
plt.ylabel('Top_5_categorical_accuracy')
plt.xlabel('epoch')
plt.grid()
plt.savefig(r'D:\code\deep_learning\Keras_practice\plot_results\inception_resnet_v2_transfer_learning\learning_{}.png'.format(NB))
plt.show()


model.save('inception_resnet_v2__11.h5')
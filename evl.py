
from keras.models import Model, Input, load_model
from keras.layers import Flatten, Activation, GlobalAveragePooling2D, Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


epochs = 50
lr_data = []

def lr_scheduler(epoch):
    if epoch > 0.9 * epochs:
        lr = 0.00005 / 8
    elif epoch > 0.75 * epochs:
        lr = 0.00005 / 4
    elif epoch > 0.5 * epochs:
        lr = 0.00005 / 2
    elif epoch > 0.25 * epochs:
        lr = 0.00005 / 2
    elif epoch > 0.10 * epochs:
        lr = 0.00005 / 2
    else:
        lr = 5e-5
    lr_data.append(lr)
    return lr

scheduler = LearningRateScheduler(lr_scheduler)

data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

vali_gen = ImageDataGenerator(
    rescale=1./255,
)

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

train_data = X_train[100:]
train_label = y_train[100:]

vali_data = X_train[:100]
vali_label = y_train[:100]


train_gen = data_gen.flow(train_data, train_label, 
                            batch_size=16)
vali_gen_data = vali_gen.flow(vali_data, vali_label, batch_size=16)

test_gen = vali_gen.flow(X_test, y_test, batch_size=16)

model = load_model('D:\\code\\deep_learning\\Keras_practice\\models\\inception_resnet_v2.h5')

model.compile(optimizer=RMSprop(lr=1e-6), loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_data)/16,
                    epochs=epochs,
                    validation_data=vali_gen_data,
                    validation_steps=len(vali_data)/16,
#                    callbacks=callbacks
                    )

print(len(model.layers))


loss_e, acc_e = model.evaluate_generator(test_gen)

print('Loss : ',loss_e, ' acc : ', acc_e)

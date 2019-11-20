import numpy as np 
import matplotlib.pyplot as plt
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import load_model

vali_gen = ImageDataGenerator(
    rescale=1./255,
)

x, y = oxflower17.load_data(one_hot=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

test_gen = vali_gen.flow(X_test, y_test, batch_size=16)

model = load_model('inception_resnet_v2.h5')

#model.summary()

loss_e, acc_e = model.evaluate_generator(test_gen)

print('[ Loss : %.4f][ acc : %.4f]'%(loss_e, acc_e))


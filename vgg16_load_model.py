import keras as K
from keras.models import load_model, save_model
from keras.applications import vgg16


# weights 會放在 " C:\Users\ib811\.keras\models "

base_model = vgg16.VGG16(include_top=False)
base_model.summary()
base_model.save('vgg16.h5')

model = load_model('vgg16.h5')
model.summary()
import os, shutil

org_dir = 'D:\\code\\deep_learning\\Keras_practice\\17flowers\\jpg'

dst_dir = r'D:\code\deep_learning\Keras_practice\flower'

data = os.listdir(org_dir+'\\0\\')

"""
for i in range(17):
    os.mkdir(dst_dir+'\\train\\{}'.format(i))
"""

for  i in range(17):
    data = os.listdir(org_dir + '\\{}\\'.format(i))
    for img in data[65:]:
        path1 = os.path.join(org_dir + '\\{}\\'.format(i), img)
        path2 = os.path.join(dst_dir + '\\test\\{}\\'.format(i), img)
        shutil.copyfile(path1, path2)



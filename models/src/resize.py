from PIL import Image
import os
import numpy as np

im = Image.open('../data/images/red_car.png')
im = im.resize((32, 32))
im = (np.array(im))

im = im[:,:,:3] / 255
im = np.expand_dims(im, axis=0)

print(im)

np.save('../data/test_sample', im)
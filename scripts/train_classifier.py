import os

import cv2

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint


def create_model():
    model = Sequential()
    model.add(Conv2D(64,3,3, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = create_model()

print(model.summary())

print("loading images")
happy_fns = [file for file in os.listdir('happy')]
nothappy_fns = [file for file in os.listdir('nothappy')]

happy_imgs = []
for happy_fn in happy_fns:
    image = cv2.imread(f"happy/{happy_fn}")
    happy_imgs.append(image)
    
nothappy_imgs = []
for nothappy_fn in nothappy_fns:
    image = cv2.imread(f"nothappy/{nothappy_fn}")
    nothappy_imgs.append(image)

print("done")

X = np.array(happy_imgs + nothappy_imgs, dtype=np.float32)
y = np.concatenate([np.ones(len(happy_imgs)), np.zeros(len(nothappy_imgs))])

X /= 255.0 # convert to float

print(X.shape, y.shape)

mcp_save = ModelCheckpoint('smile_detector_twitch.h5',
                           save_best_only=True,
                           monitor='val_accuracy',
                           verbose=1,
                           mode='max')

model.fit(X, y,
          epochs=50,
          batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[mcp_save])


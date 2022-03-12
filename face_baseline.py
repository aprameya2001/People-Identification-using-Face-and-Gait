import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from sklearn.utils import shuffle

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

x = []
y = []

for i in range(40):
    for j in range(10):
        path = 'extracted_dataset' + '/' + 'person' + str(i+1) + '/' + 'face' + '/' + str(j+1) + '.pgm'
        image = cv2.imread(path, -1)
        image = cv2.resize(image, (12, 12))
        image = image.reshape(12, 12, 1)

        x.append(image)
        y.append([i])


x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
y = to_categorical(y)

x, y = shuffle(x, y)

model = Sequential()
model.add(Conv2D(24, kernel_size=3, activation='relu', input_shape=(12, 12, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(40, activation='softmax'))
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

EPOCHS = 5
BATCH_SIZE = 1
model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split = 0.3)
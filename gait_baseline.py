import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from sklearn.utils import shuffle

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ConvLSTM2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

X = []
Y = []
C = 40

for i in range(C):
    gait_folder = 'extracted_dataset' + '/' + 'person' + str(i+1) + '/' + 'gait' + '/'
    styles = os.listdir(gait_folder)
    styles.sort()
    for j, style in enumerate(styles):
        style_folder = gait_folder + style + '/'
        angles = os.listdir(style_folder)
        angles.sort()
        x = []
        for angle in angles:
            angle_folder = style_folder + angle + '/'
            images = os.listdir(angle_folder)
            images.sort()
            xx = []
            for image in images:
                path = angle_folder + image
                img = cv2.imread(path, -1)
                img = cv2.resize(img, (64, 64))
                img = np.array(img, dtype=np.float32)
                xx.append(img)
            xx = np.array(xx, dtype=np.float32)
            x.append(xx)
        x = np.stack(x, axis=3)
        X.append(x)
        Y.append(i)
        print('person', str(i+1), 'style', str(j+1))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
Y = to_categorical(Y)

X, Y = shuffle(X, Y)

model = Sequential()
model.add(ConvLSTM2D(32, kernel_size = 3, input_shape=(32, 64, 64, 11), return_sequences=True))
model.add(Dropout(0.5))
model.add(ConvLSTM2D(16, kernel_size=3, return_sequences=True))
model.add(Dropout(0.5))
model.add(ConvLSTM2D(8, kernel_size=3, return_sequences=True))
model.add(Dropout(0.5))
model.add(ConvLSTM2D(4, kernel_size=3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(C, activation='softmax'))
model.compile(optimizer = Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

EPOCHS = 10
BATCH_SIZE = 4
model.fit(X, Y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.3, validation_batch_size = BATCH_SIZE, shuffle = True)
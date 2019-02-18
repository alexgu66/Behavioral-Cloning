
import csv
import cv2
import numpy as np

adjust = 0.2
epochs = 200
batch = 128

samples = []
with open('./data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

# Use generator to save memory usage
def generator(samples, batch_size=32):
    num_samples = len(samples) 
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]       

            images = []
            measurements = []
            for line in batch_samples:
                # add center, left, right and flip image 
                for i_camera in range(3):
                    source_path = line[i_camera]
                    filename = source_path.split('/')[-1]
                    current_path = './data3/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image, 1))
                measurement = float(line[3])
                # add add center, left(+adjust), right(-adjust) and flip angle
                measurements.append(measurement)
                measurements.append(measurement * -1.0)
                measurements.append(measurement + adjust)
                measurements.append((measurement + adjust) * -1.0)
                measurements.append(measurement - adjust)   
                measurements.append((measurement - adjust) * -1.0)                                
            
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch)
validation_generator = generator(validation_samples, batch_size=batch)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# use Nivida model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# register 2 callbacks, early stop when val los change is small
# and save model after each epoch
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=0, mode='auto'),
             ModelCheckpoint('./epoch_{epoch:03d}-loss_{loss:.5f}_val_loss_{val_loss:.5f}.h5', \
            monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, \
            mode='auto', period=1)]

history_object = model.fit_generator(train_generator, samples_per_epoch=\
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1, callbacks=callbacks)

model.save('model5.h5')

from keras.models import Model
import matplotlib.pyplot as plt

# ## print the keys contained in the history object
print(history_object.history.keys())

# ## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')

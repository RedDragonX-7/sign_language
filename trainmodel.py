from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import TensorBoard

import numpy as np
import os

#map actions to numerical values
label_map = {label:num for num, label in enumerate(actions)}

# load and pad sequences

sequences, labels = [], []
for action in actions:
    for sequence_num in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence_num), f'{frame_num}.npy')
            
            if os.path.exists(npy_path):
                res = np.load(npy_path,allow_pickle=True)
                window.append(res)
            else:
                window.append(np.zeros(21*3))

        sequences.append(window)
        labels.append(label_map[action])

# prepare the training and vaoidation sets
x = np.array(sequences)        
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, stratify=y)

# saving the logs for teansorboard
log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)

# define a model
model = Sequential([
    Input(shape=(sequence_length, 21*3)),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# train the model
model.fit(x_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(x_test, y_test))
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
print('Model saved successfully')
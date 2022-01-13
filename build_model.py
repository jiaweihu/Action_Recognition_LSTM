from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from generate_model_data import model_data
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class build_lstm:
    def __init__(self):
        # self.lstm_model = lstm_model()
        self.label_map = {label:num for num, label in enumerate(model_data.actions)}
        self.sequences, self.labels = [], []
        for action in model_data.actions:
            for sequence in range(model_data.no_sequences):
                window = []
                for frame_num in range(model_data.sequence_length):
                    res = np.load(os.path.join(model_data.data_path, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                self.sequences.append(window)
                self.labels.append(self.label_map[action])

    def build(self):
        X = np.array(self.sequences)
        y = to_categorical(self.labels).astype(int)     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(model_data.actions.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
        model.summary()

        model.save('model/action.h5')
        del model

    def run(self):
        self.build()

if __name__ == '__main__':
    func = build_lstm()
    func.run()


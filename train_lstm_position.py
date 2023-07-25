""""
Usage
    python3 train_lstm_position.py path_to_dataset_dir input_seq_len output_seq_len
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time  # helper libraries
import os
import sys
import pickle  # import the pickle module

# convert an array of values into a dataset matrix
def create_dataset(dataset, input_seq_len, output_seq_len):
    dataX, dataY = [], []
    for i in range(len(dataset) - input_seq_len - output_seq_len):
        a = dataset[i:(i + input_seq_len), :]
        dataX.append(a)
        b = dataset[i + input_seq_len : (i + input_seq_len + output_seq_len), :]
        dataY.append(b.flatten())  # flatten the output sequences
    return np.array(dataX), np.array(dataY)

# load the dataset
dir_path = sys.argv[1]
input_seq_len = int(sys.argv[2])  # for example: 10
output_seq_len = int(sys.argv[3])  # for example: 5

all_files = os.listdir(dir_path)
csv_files = [f for f in all_files if f.endswith('.csv') or f.endswith('.txt')]

# Prepare dataset to compute the scaler
dataframes = []
for file in csv_files:
    file_path = os.path.join(dir_path, file)
    df = pd.read_csv(file_path, usecols=['tx', 'ty', 'tz'])
    dataframes.append(df)
all_data = pd.concat(dataframes).values.astype('float32')

# Compute the scaler on all data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data)

# create the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()

# Add first GRU layer
# return_sequences=True is necessary for stacking GRU layers
# Add first LSTM layer
model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=(input_seq_len, 3)))) 
# Add 2nd LSTM layer
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.1))
model.add(Dense(output_seq_len*3))
model.compile(loss='mean_squared_error', optimizer='adam')

# Record the start time
start_time = time.time()

# Process each file
for file in csv_files:
    file_path = os.path.join(dir_path, file)
    print(f"Using trajectory file: {file_path}\n")
    df = pd.read_csv(file_path, usecols=['tx', 'ty', 'tz'])
    dataset = df.values.astype('float32')

    # normalize the dataset using the scaler computed on all data
    dataset = scaler.transform(dataset)

    # split into train and test sets, 67% for training and the rest 33% for testing
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1, timestep 240
    trainX, trainY = create_dataset(train, input_seq_len, output_seq_len)
    testX, testY = create_dataset(test, input_seq_len, output_seq_len)

    history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Compute RMSE
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print(f'Train Score for {file}: {trainScore:.2f} RMSE')
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print(f'Test Score for {file}: {testScore:.2f} RMSE')

# Calculate the computation time
end_time = time.time()
computation_time = end_time - start_time

print(f"Training time: {computation_time} seconds\n")

# Save the model for future use
model_file = 'drone_position_prediction_model.h5'
model.save(model_file)
print(f"Model trained and saved at {model_file}")

# Save the scaler for future use
scaler_file = 'scaler_position.pkl'
pickle.dump(scaler, open(scaler_file, 'wb'))
print(f"Model scaler saved at {scaler_file}")

# Plot the loss history
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Add the computation time on the plot
plt.text(0, max(history.history['loss']), f'Computation time: {computation_time} seconds')

plt.show()

"""
Usage
    python3 predict_position.py path_to_model path_to_scaler path_to_csv sequence_index input_seq_len output_seq_len
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import load_model
import sys
from sklearn.preprocessing import MinMaxScaler
import pickle

# load the model
model = load_model(sys.argv[1])
# Load the scaler
scaler = pickle.load(open(sys.argv[2], 'rb'))

# load the original data
dataframe = pd.read_csv(sys.argv[3])
dataset = dataframe[['tx', 'ty', 'tz']].values  # select 'tx', 'ty', 'tz' columns
dataset = dataset.astype('float32')

# normalizing the dataset
dataset = scaler.transform(dataset)
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, forecast_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-forecast_steps):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        b = dataset[(i+look_back):(i+look_back+forecast_steps), :]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = int(sys.argv[5])
forecast_steps = int(sys.argv[6])
testX, testY = create_dataset(dataset, look_back, forecast_steps)

# select the sequence to visualize
sequence_index = int(sys.argv[4])
example_sequence = testX[sequence_index]
actual_output = testY[sequence_index]

# make a prediction with the selected sequence
predicted_output = model.predict(example_sequence[np.newaxis, :, :])

# reshape the sequence and outputs to their original shapes
example_sequence = example_sequence.reshape(-1, 3)
predicted_output = predicted_output.reshape(-1, 3)
actual_output = actual_output.reshape(-1, 3)

# inverse transform
example_sequence = scaler.inverse_transform(example_sequence)
predicted_output = scaler.inverse_transform(predicted_output)
actual_output = scaler.inverse_transform(actual_output)
full_trajectory = scaler.inverse_transform(dataset)

# compute the Euclidean distance
diff = predicted_output - actual_output
dist = np.linalg.norm(diff)
print(f"The Euclidean distance between the predicted and true sequences is {dist:.2f}")

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the example sequence
ax.plot(example_sequence[:, 0], example_sequence[:, 1], example_sequence[:, 2], 'b', label='Input Sequence')

# plot the predicted output
ax.plot(predicted_output[:, 0], predicted_output[:, 1], predicted_output[:, 2], 'r', label='Predicted Output')

# plot the entire true trajectory
# ax.scatter(full_trajectory[:, 0], full_trajectory[:, 1], full_trajectory[:, 2], c='g', label='Full True Trajectory')

# plot the predicted output
ax.plot(actual_output[:, 0], actual_output[:, 1], actual_output[:, 2], 'k', label='actual Output')

ax.set_xlabel('tx')
ax.set_ylabel('ty')
ax.set_zlabel('tz')

# Display the legend
ax.legend()

plt.show()


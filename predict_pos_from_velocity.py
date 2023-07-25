"""
Usage
    python3 predict_pos_from_velocity.py path_to_dataset_file model_path scaler_path sequence_index input_seq_len output_seq_len dt
"""
import sys
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_velocity(position_data, dt):
    velocity_data = np.diff(position_data, axis=0) / dt
    return velocity_data


def integrate_velocity(current_position, velocity, dt):
    future_position = current_position + velocity * dt
    return future_position


def predict_future_positions(file_path, model_path, scaler_path, sample_index, input_seq_len, output_seq_len, time_step=1):
    # Load position data
    position_data = pd.read_csv(file_path, usecols=['tx', 'ty', 'tz']).values.astype('float32')

    # Check if the sample_index is valid
    if sample_index < 0 or sample_index >= len(position_data) - 1 - input_seq_len - output_seq_len:
        raise ValueError('Invalid sample_index. It should be in the range [0, len(position_data) - 1 - input_seq_len - output_seq_len).')

    # Compute velocity data from position data
    velocity_data = compute_velocity(position_data, time_step)

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Normalize velocity data
    velocity_data_norm = scaler.transform(velocity_data)

    # Get the specific sample for prediction
    sample = velocity_data_norm[sample_index:sample_index+input_seq_len]

    # Reshape sample for model input, shape (1, sequence_length, n_features)
    sample = np.reshape(sample, (1, -1, 3))

    # Load the trained model
    model = load_model(model_path)

    # Predict future velocities
    future_velocities_norm = model.predict(sample)

    # Reshape to (output_seq_len, 3)
    future_velocities_norm = future_velocities_norm.reshape(output_seq_len, 3)

    # Denormalize the future velocities
    future_velocities = scaler.inverse_transform(future_velocities_norm)

    # Reshape back to (1, output_seq_len*3)
    future_velocities = future_velocities.reshape(1, output_seq_len*3)

    # Initialize the list of future positions
    future_positions = [position_data[sample_index + input_seq_len]]

    # Calculate future positions by integrating the velocities
    for v in future_velocities[0].reshape(output_seq_len, 3):
        future_position = integrate_velocity(future_positions[-1], v, time_step)
        future_positions.append(future_position)

    # Get the input and true output positions
    input_positions = position_data[sample_index:sample_index+input_seq_len]
    true_output_positions = position_data[sample_index+input_seq_len:sample_index+input_seq_len+output_seq_len]

    # Plot the positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the input positions
    ax.plot(input_positions[:, 0], input_positions[:, 1], input_positions[:, 2], 'b-', label='Input Positions')

    # Plot the true output positions
    ax.plot(true_output_positions[:, 0], true_output_positions[:, 1], true_output_positions[:, 2], 'g-', label='True Output Positions')

    # Plot the predicted positions
    future_positions = np.array(future_positions)
    ax.plot(future_positions[:, 0], future_positions[:, 1], future_positions[:, 2], 'r-', label='Predicted Positions')

    ax.legend()
    plt.show()

    # Return the future positions
    return future_positions


if __name__ == "__main__":
    file_path = sys.argv[1]
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]
    sample_index = int(sys.argv[4])
    input_seq_len = int(sys.argv[5])
    output_seq_len = int(sys.argv[6])
    dt = float(sys.argv[7])

    predict_future_positions(file_path, model_path, scaler_path, sample_index, input_seq_len, output_seq_len, dt)

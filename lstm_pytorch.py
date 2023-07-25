import sys
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""
usage
    python lstm_pytorch.py path_to_your_file.csv 10 5
"""
# Read the file path from command-line arguments
file_path = sys.argv[1]
sequence_length = int(sys.argv[2])
forecast_length = int(sys.argv[3])

# Load and preprocess data
df = pd.read_csv(file_path)
scaler = MinMaxScaler()
data = scaler.fit_transform(df[['tx', 'ty', 'tz']])

# Split into training and test datasets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)

# Function to create a dataset to feed into an LSTM
def create_inout_sequences(input_data, tw, forecast):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw-forecast):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+forecast]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=forecast_length*3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Hyperparameters
input_size = 3
hidden_layer_size = 100
output_size = forecast_length*3
epochs = 100

# Initialize model
model = LSTM(input_size, hidden_layer_size, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data sequences
tw = sequence_length  # Sequence length
forecast = forecast_length  # Number of steps to forecast
train_inout_seq = create_inout_sequences(train_data, tw, forecast)
test_inout_seq = create_inout_sequences(test_data, tw, forecast)

# Train model
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Make predictions
fut_pred = forecast_length
test_inputs = test_inout_seq[:fut_pred]
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[i][0])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq))

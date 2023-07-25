# lstm_trajectory_prediction
This repo provides scripts to train and test LSTM models for position trajectory prediciton

# Prerequisites
* keras/tensorflow
* pandas
* matplotlib
* numpy

# Training
* To train an LSTM model on position trajecetories, use
    ```bash
    python3 train_lstm_position.py path_to_dataset_dir input_seq_len output_seq_len
    ```

* To train an LSTM model on velocity trajecetories estimated from position trajectories, use
    ```bash
    python3 train_lstm_velocity.py path_to_dataset_dir input_seq_len output_seq_len
    ```
* Each script will generate a `.h5` file of the trained model, and the scaler file `*.pkl`

# Prediction
* To predict using model trained on position data, use
    ```bash
    python3 predict_position.py path_to_model path_to_scaler path_to_csv sequence_index input_seq_len output_seq_len
    ```

* To predict using the model trained on velocity data, use
    ```bash
    python3 predict_pos_from_velocity.py path_to_dataset_file model_path scaler_path sequence_index input_seq_len output_seq_len dt
    ```
* Arguments
    * `dt` is the sampling time that is used in the dataset in seconds
    * `sequence_index` is the input sequence index in the dataset to use for prediction
    * `input_seq_len` The length of the input sequence which is the number of points of the input sequence
    * `output_seq_len` The length of the output sequence which is the number of points of the output sequence

# Notes
`lstm_pytorch` is not tested, and not workgin!
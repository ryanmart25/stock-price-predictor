# Stock Price Predictor Report

## Problem Statement

The task was creating three models with varying architectures with the goal of training them to predict the stock price of 1 company based on the last 7 day's of stock data. TO accomplish this Fully Connected, Convolutional, and Recurrent models were designed and created.

## Methodology

For all models, the data had to be reshaped to be compatible with the requirements of using the last 7 days to predict tomorrow's stock price and normalized. There were no categorical features, so all features were simply z-score encoded.

For the FCN, re-shaping the data involved forming each record such that it was 7 days * 5 feature columns, which yielded an array of 35 numbers for each record in the training and test datasets. For the CNN, I tried two different input shapes. I formed the data into a two-dimensional image with seven rows and 5 columns, each pixel with 1 channel, as well as a 1 dimensional image of 7 pixels, each pixel with 5 channels. For the RNN, the data was shaped into sequences of 7 vectors, each with 5 numbers. 

## Experimental REsults and Analysis

My best models were the CNN and FCN. They had a RMSE of 1.24 and 1.09 respectively. The best RNN model had a RMSE of  8.5. All models used Early Stopping, Model Checkpoint, and Adam as their optimizer. I chose mean squared error as the loss function for all models. The FCN had the following structure:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_84 (InputLayer)          │ (None, 35)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_192 (Dense)                    │ (None, 256)                 │           9,216 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_193 (Dense)                    │ (None, 256)                 │          65,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_194 (Dense)                    │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_195 (Dense)                    │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_196 (Dense)                    │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_197 (Dense)                    │ (None, 16)                  │             528 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_198 (Dense)                    │ (None, 8)                   │             136 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_199 (Dense)                    │ (None, 4)                   │              36 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_200 (Dense)                    │ (None, 1)                   │               5 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

While the CNN had the following structure:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_90 (InputLayer)          │ (None, 7, 5)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_12 (Conv1D)                   │ (None, 7, 64)               │             384 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_13 (Conv1D)                   │ (None, 7, 64)               │          12,352 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d_6 (MaxPooling1D)       │ (None, 3, 64)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_7 (Flatten)                  │ (None, 192)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_211 (Dense)                    │ (None, 32)                  │           6,176 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_212 (Dense)                    │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
While the RNN had the following structure: 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_68 (InputLayer)          │ (None, 7, 5)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_110 (LSTM)                      │ (None, 70)                  │          21,280 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_132 (Dense)                    │ (None, 16)                  │           1,136 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_133 (Dense)                    │ (None, 1)                   │              17 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
The lift charts for all three models in order follow: 
![FCN Lift Chart - iteration 57](output\iteration-57\FCN-Lift-Chart.png "FCN Lift Chart")
![FCN Lift Chart - iteration 57](output\iteration-57\CNN-Lift-Chart.png "FCN Lift Chart")
![FCN Lift Chart - iteration 57](output\iteration-53\RNN-Lift-Chart.png "FCN Lift Chart")

## Task Division and Project Reflection

I worked by myself for this project. In reflection, it was interesting to see how the same problem could be solved with three different architectures, and how each architecture required its own process for hyper parameter tuning. If I had more time I would like to further develop the RNN model. In particular, I'd like to figure out why adding more LSTM layers made the model perform worse, why a single large LSTM layer performed better than numerous small layers, and why 1 small dense layer between the output and LSTM layer is needed. I also want to know why the SGD optimizer always led to worse performance.

## Additional Features

I did not try to implement additional features.

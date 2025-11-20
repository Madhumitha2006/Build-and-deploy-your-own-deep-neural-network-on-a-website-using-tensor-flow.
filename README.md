# Build-and-deploy-your-own-deep-neural-network-on-a-website-using-tensor-flow.
## Name: Madhu Mitha V
## Register Number: 2305002013
## AIM
To build a deep neural network (DNN) model using TensorFlow, convert the model into TensorFlow.js format, and deploy it on a website so that predictions can be performed directly inside a web browser.

## ALGORITHM
1.Prepare the dataset and split it into training and testing sets.

2.Build the deep neural network model using TensorFlow/Keras with multiple Dense layers.

3.Train and evaluate the model using training data.

4.Save and convert the trained model into TensorFlow.js format using the TFJS converter.

5.Create a web page (HTML + JavaScript) and load the TensorFlow.js model.

6.Perform prediction inside the browser using user input and display the output on the webpage.

## PROGRAM
```

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("TensorFlow version:", tf.__version__)

# Step 1: Prepare a simple dataset
# Example task: Predict the sum of two numbers
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8]
], dtype=float)

y = np.array([3, 5, 7, 9, 11, 13, 15], dtype=float)

# Step 2: Build the Deep Neural Network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 4: Train the model
print("\nTraining the model...")
history = model.fit(X, y, epochs=500, verbose=0)

print("Training complete!")

# Step 5: Test the model with a new input
test_data = np.array([[10, 20]], dtype=float)
predicted_value = model.predict(test_data)

print("\nInput  : [10, 20]")
print("Output :", predicted_value[0][0])

# Step 6: Save the model for deployment
model.save("dnn_sum_model.h5")
print("\nModel saved as dnn_sum_model.h5")


```
## OUTPUT
<img width="671" height="238" alt="image" src="https://github.com/user-attachments/assets/b4d16d1f-80ff-491b-a2d8-04c122703826" />


## RESULT
A deep neural network model was created using TensorFlow/Keras, trained, and successfully converted to TensorFlow.js format.
The model was deployed on a web browser where predictions were computed in real time using JavaScript.
This demonstrates that machine learning models can be deployed on websites without requiring a backend server.

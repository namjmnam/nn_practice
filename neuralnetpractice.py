from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

data = [
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
    ([215, 745], 102),
    ([1110, 125], 127),
]

def preprocess_data(data):
    # Assuming data is a list of ((input1, input2), output)
    X = []  # Inputs
    y = []  # Outputs

    for ((num1, num2), sum_) in data:
        # Break down each number into its digits
        X.append(
            [int(d) for d in str(num1).zfill(4)] +  # Pad with zeros to ensure 4 digits
            [int(d) for d in str(num2).zfill(4)]
        )
        # Process output similarly but ensure it's 4 digits (truncate if necessary)
        y.append(
            [int(d) for d in str(sum_).zfill(4)[:4]]
        )

    # Normalize digits to be between 0 and 1
    X_normalized = np.array(X) / 9.0  # Since 9 is the max digit value
    y_normalized = np.array(y) / 9.0

    return X_normalized, y_normalized

X_normalized, y_normalized = preprocess_data(data)
# print(X_normalized, '\n\n', y_normalized)

# Assuming your inputs are normalized and ready
input_layer = Input(shape=(8,))  # Combined input for both numbers, each represented by 4 digits
fc1 = Dense(128, activation='relu')(input_layer)
fc2 = Dense(64, activation='relu')(fc1)
output_layer = Dense(4, activation='sigmoid')(fc2)  # Predicting 4 normalized digits

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_normalized, y_normalized, epochs=100, batch_size=32, validation_split=0.2)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

def preprocess_custom_data(num1, num2):
    # Convert numbers to their digit representation, normalized
    digits_num1 = [int(d) for d in str(num1).zfill(4)]  # Ensure it is a 4-digit number
    digits_num2 = [int(d) for d in str(num2).zfill(4)]
    combined = digits_num1 + digits_num2  # Combine the digit representations
    X_custom = np.array([combined]) / 9.0  # Normalize and ensure it's a 2D array for Keras
    return X_custom

# 입력인자
num1 = 655
num2 = 950

# Preprocess the custom data
X_custom_normalized = preprocess_custom_data(num1, num2)

# Predict the output using the model
prediction_normalized = model.predict(X_custom_normalized)

# Assuming the output is normalized, denormalize it if necessary
prediction = prediction_normalized * 9.0  # Denormalize based on how the output was normalized

# Convert predicted digits back to a single number, if necessary
predicted_digits = np.round(prediction[0] * 9).astype(int)  # Assuming you want to round to nearest integer
predicted_number = int(''.join(map(str, predicted_digits)))

print("Predicted output:", predicted_number)


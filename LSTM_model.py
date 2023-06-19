import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Define the value of Future (modify according to your needs)
Future = 1

# Load the trained LSTM model
model_path = './model/Pump_LSTM_Fapi_4_' + str(Future) + '.h5'
model = load_model(model_path)

# Define the Streamlit app
def main():
    st.title("LSTM-based System")
    st.write("Enter the sensor data:")

    # Create input fields for sensor data
    sensor1 = st.number_input("Sensor 1")
    sensor2 = st.number_input("Sensor 2")
    sensor3 = st.number_input("Sensor 3")
    sensor4 = st.number_input("Sensor 4")

    # Preprocess the user-provided data
    user_data = np.array([[sensor1, sensor2, sensor3, sensor4]])
    num_features = 45

    # Check if padding is required
    if user_data.shape[1] < num_features:
        # Calculate padding width
        padding_width = num_features - user_data.shape[1]

        # Pad the input data with zeros
        user_data = np.pad(user_data, ((0, 0), (0, padding_width)), mode='constant')

    # Reshape the input data
    user_data = np.reshape(user_data, (user_data.shape[0], 1, user_data.shape[1]))

    # Make predictions using the loaded model
    prediction = model.predict(user_data)
    predicted_class = np.argmax(prediction[1][0])

    # Map class label to name
    class_names = ["Broken", "Normal", "Recovering"]
    predicted_class_name = class_names[predicted_class]

    # Display the predicted signal and class
    st.write("Predicted Class:", predicted_class_name)

# Run the Streamlit app
if __name__ == "__main__":
    main()


import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import librosa
from twilio.rest import Client
import tkinter as tk
from tkinter import messagebox
import threading
import keras

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import models, layers
from keras.models import load_model

# Twilio SMS function
def send_sms(to_number, message_body):
    try:
        account_sid = 'ACb3352027ff10eacadc0df9f68dff6445'
        auth_token = 'f351900136f69e4b84c5d574154c5a0f'
        twilio_number = '+14159497179'
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=to_number
        )
        print(f"SMS sent to {to_number}: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Function to extract features from the audio
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {e}")
        return None

# Function to predict scream (using your pre-trained model)
def predict_scream(file, model):
    mfccs = extract_features(file)
    if mfccs is not None:
        mfccs = mfccs.reshape(1, mfccs.shape[0], 1, 1)
        prediction = model.predict(mfccs)
        if prediction > 0.5:
            return True
        else:
            return False
    return None

# Function to record voice
def record_voice(filename='output.wav', duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    voice_data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, voice_data)
    print(f"Recording saved to {filename}")

# GUI functionality
def start_recording():
    record_voice('output.wav')
    status_label.config(text="Audio recorded, predicting...")
    threading.Thread(target=process_audio).start()

def process_audio():
    result = predict_scream('output.wav', model)
    if result:
        status_label.config(text="Scream detected! Sending SMS...")
        message = "Emergency! Threat detected."
        for contact in emergency_contacts:
            send_sms(contact, message)
        messagebox.showwarning("Threat Detected", "Scream detected! SMS sent to emergency contacts.")
    else:
        status_label.config(text="No scream detected.")

# GUI setup
root = tk.Tk()
root.title("Scream Detection System")

# Labels and buttons
status_label = tk.Label(root, text="Press 'Record' to start", font=("Helvetica", 14))
status_label.pack(pady=20)

record_button = tk.Button(root, text="Record", font=("Helvetica", 14), command=start_recording)
record_button.pack(pady=20)

# Emergency contacts
emergency_contacts = ["+916005748700", "+917006902591"]
# Path to your dataset (make sure to update with correct paths)
scream_files = [f'C:\\Users\\Dell\\Desktop\\fowzan\\Converted_Separately\\scream\\{file}' for file in os.listdir(r'C:\Users\Dell\Desktop\fowzan\Converted_Separately\scream') if file.endswith('.wav')]
non_scream_files = [f'C:\\Users\\Dell\\Desktop\\fowzan\\Converted_Separately\\non_scream\\{file}' for file in os.listdir(r'C:\Users\Dell\Desktop\fowzan\Converted_Separately\non_scream') if file.endswith('.wav')]


# Extract features and labels
features = []
labels = []

# Process scream files
for file in scream_files:
    print(f"Processing file: {file}")
    mfccs = extract_features(file)
    if mfccs is not None:
        features.append(mfccs)
        labels.append('scream')
    else:
        print(f"Skipping file: {file} due to extraction error.")

# Process non-scream files
for file in non_scream_files:
    print(f"Processing file: {file}")
    mfccs = extract_features(file)
    if mfccs is not None:
        features.append(mfccs)
        labels.append('non_scream')
    else:
        print(f"Skipping file: {file} due to extraction error.")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape the data for CNN input (add 2 more dimensions)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Build the CNN model
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (scream or non-scream)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and save the model
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])
model = build_model(input_shape)

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('scream_detection_model.h5')

# Load the pre-trained model for real-time interaction
model = load_model('scream_detection_model.h5')

# Start interaction with user


# Load your pre-trained model here
#model = load_your_model_function()  # Add your model loading function

# Start GUI loop
root.mainloop()

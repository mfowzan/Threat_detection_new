import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from twilio.rest import Client
import librosa
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

# Function to extract features from audio
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {e}")
        return None

# Function to predict scream using pre-trained model
def predict_scream(file, model):
    mfccs = extract_features(file)
    if mfccs is not None:
        mfccs = mfccs.reshape(1, mfccs.shape[0], 1, 1)
        prediction = model.predict(mfccs)
        if prediction > 0.5:  # Assuming > 0.5 means "scream detected"
            return True
        else:
            return False
    else:
        return None

# Record voice in real-time
def record_voice(filename='output.wav', duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    voice_data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, voice_data)  # Save as WAV file
    print(f"Recording saved to {filename}")

# Real-time interaction loop with user
def interact_with_user(model, emergency_contacts):
    while True:
        print("\nPress 'r' to record a 5-second audio sample or 'q' to quit:")
        user_input = input().lower()
        if user_input == 'r':
            voice_file = 'output.wav'
            record_voice(voice_file)
            
            # Step 1: Predict if scream is detected
            result = predict_scream(voice_file, model)
            if result == True:
                print("Threat detected! Sending SMS to emergency contacts.")
                message = "Emergency! Threat detected. Please take immediate action."
                for contact in emergency_contacts:
                    send_sms(contact, message)
            elif result == False:
                print("No scream detected.")
            else:
                print("Error in processing audio.")
        elif user_input == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please press 'r' to record or 'q' to quit.")

# Replace with actual emergency contacts (user-defined)
emergency_contacts = [
    "+916005748700",
    "+917006902591"
]

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
interact_with_user(model, emergency_contacts)

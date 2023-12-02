import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('/content/model1.h5')

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Open a connection to the webcam (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Preprocess the frame for prediction
    img_for_prediction = preprocess_image(frame)

    # Make a prediction
    prediction = model.predict(img_for_prediction)

    # Accessing individual probabilities for binary classification
    prob_class_0 = prediction[0][0]
    prob_class_1 = prediction[0][1]

    # Print the probabilities
    print("Probability for Class 0:", prob_class_0)
    print("Probability for Class 1:", prob_class_1)

    # Perform actions based on the predictions
    if prob_class_0 > 0.5:
        print("Class 0 detected")
        # Perform actions for Class 0

    if prob_class_1 > 0.5:
        print("Class 1 detected")
        # Perform actions for Class 1

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

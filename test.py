import cv2
import numpy as np
from pathlib import Path
from concrete.ml.common.serialization.loaders import load

# Load the saved Concrete ML model
model_path = Path("./saved_model_direct.json")
with model_path.open("r") as f:
    model = load(f)

# Compile the model (required after loading)
# Replace `150` with the correct number of features used during training
sample_input = np.random.rand(1, 150)  # Correct number of features
model.compile(sample_input)

print("Model loaded and compiled successfully!")

# Define the image dimensions (must match the training data dimensions)
IMAGE_HEIGHT = 15  # Replace with the height of your training images
IMAGE_WIDTH = 10   # Replace with the width of your training images

# Define the target names (must match the training labels)
target_names = [
    "Ariel Sharon",
    "Colin Powell",
    "Donald Rumsfeld",
    "George W Bush",
    "Gerhard Schroeder",
    "Hugo Chavez",
    "Tony Blair",
]

def preprocess_image(image):
    """
    Preprocess the input image to match the model's input format.
    - Resize the image to the required dimensions.
    - Convert it to grayscale.
    - Flatten the image into a 1D array.
    """
    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    flattened_image = grayscale_image.flatten() / 255.0  # Normalize pixel values
    return flattened_image

def predict_president(image):
    """
    Predict the president in the given image using the loaded model.
    """
    preprocessed_image = preprocess_image(image)
    prediction = model.predict([preprocessed_image])  # Model expects a 2D array
    predicted_label = int(prediction[0])
    return target_names[predicted_label]

# Initialize OpenCV video capture (use 0 for the default camera)
cap = cv2.VideoCapture(0)

print("Starting video capture. Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Predict the president in the frame
    try:
        predicted_name = predict_president(frame)
        cv2.putText(
            frame,
            f"Predicted: {predicted_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    except Exception as e:
        cv2.putText(
            frame,
            "Prediction failed!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        print(f"Error during prediction: {e}")

    # Display the frame with the prediction
    cv2.imshow("President Detector", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
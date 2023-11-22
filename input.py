from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model architecture from the JSON file
with open("modelGG.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights("model66.h5")

# Compile the model (you might need to specify the optimizer and loss function)
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the input image
img_path = r"C:\Users\R S PARATE\Downloads\download.jfif"  # Use raw string to handle backslashes
img = image.load_img(img_path, target_size=(300, 300), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values

# Make predictions
predictions = loaded_model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(predictions)

# Print the result
print(f"Predicted class: {predicted_class}")

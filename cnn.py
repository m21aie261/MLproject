import tensorflow as tf
from tensorflow.keras import layers, models

# CNN


# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Flatten the output before the fully connected layers
        layers.Flatten(),
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Define input shape and number of classes
input_shape = (150, 150, 3)  # Example input shape for RGB images
num_classes = 10              # Example number of classes

# Create the CNN model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

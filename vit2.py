import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_classes = 10
input_shape = (32,32,3)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 40
image_size = 72  # Resize input images to this size
patch_size = 6   # Size of the patches to extract from the input images
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim*2, projection_dim]  # Transformer layers size
transformer_layers = 8
mlp_heads_units = [2048, 1024]  # Dense layers for the final classifier

# Data Augmentation pipeline
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
)
data_augmentation.layers[0].adapt(x_train)

# Multi-layer Perceptron function
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Define patches layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Display a random image from x_train
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image)
plt.axis("off")

# Resizing and extracting patches
resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))
patches_layer = Patches(patch_size)
patches = patches_layer(resized_image)

print(f"Image size: {image_size} x {image_size}")
print(f"Patch size: {patch_size} x {patch_size}")
print(f"Shape of patches: {patches.shape}")

# Build the Transformer model
inputs = layers.Input(shape=(image_size, image_size, 3))

# Apply data augmentation
x = data_augmentation(inputs)

# Extract patches from the image
x = Patches(patch_size)(x)

# Transformer layers
for _ in range(transformer_layers):
    x_residual = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([x_residual, x])

    # Feed-forward network
    x_residual = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(transformer_units[1], activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([x_residual, x])

# MLP classification head
x = layers.GlobalAveragePooling1D()(x)
x = mlp(x, mlp_heads_units, dropout_rate=0.5)

# Output layer
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Define the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.summary()
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Print dataset shapes
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


# Data preprocessing (fixed redundancy)
x_train = x_train[:500]
x_test = x_test[:500]
y_test = y_test


# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epoches = 40
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extracted from the input layers
num_patches = (image_size // patch_size) ** 2
projection_dim =64
num_heads = 4
transformer_units =[
    projection_dim*2,
    projection_dim
]# size of the transformer layers
transformer_layer = 8
mlp_heads_units = [2048,1024]# size of the dense layers of the final classification
# Data augmentation (resize and flip)
data_augmentation = keras.Sequential(
  [
    layers.Normalization(),
    layers.Resizing(image_size, image_size),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=0.02),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2)
  ],
)
data_augmentation.layers[0].adapt(x_train)

# MLP function (for transformer later)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patch extraction layer
class PatchExtractor(layers.Layer):  # Rename the class to 'PatchExtractor'
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
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


# Display one image
plt.figure(figsize=(4,4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

# Resize the image
resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))

# Extract patches
patch_layer = PatchExtractor(patch_size)
extracted_patches = patch_layer(resized_image)

# Print image details
print(f"Image size: {image_size} x {image_size}")
print(f"Patch size: {patch_size} x {patch_size}")
print(f"Patches per image: {extracted_patches.shape[1]}")
print(f"Elements per patch: {extracted_patches.shape[-1]}")

# Plot patches
n = int(np.sqrt(extracted_patches.shape[1]))  # Number of rows and columns
plt.figure(figsize=(6,6))
for i, patch in enumerate(extracted_patches[0]):
    ax = plt.subplot(n, n, i+1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))  # Reshape to (6, 6, 3)
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder,self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )
    def call(self,patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)

    patch_extractor = PatchExtractor(patch_size)
    patches = patch_extractor(augmented)

    # Make sure to pass the correct arguments to PatchEncoder
    patch_encoder = PatchEncoder(num_patches, projection_dim)
    encoded_patches = patch_encoder(patches)

    for _ in range(transformer_layer):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_heads_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate = learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
    keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
]

    )
    checkpoint_filepath = "./tmp/checpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epoches,
        validation_split=0.1,
        callbacks=[checkpoint_callback],


    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top-5 accuracy: {round(top_5_accuracy * 100, 2)}%")


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)










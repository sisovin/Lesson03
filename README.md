# Lesson03: How to work with TensorFlow

A concise summary of the code steps:

1. Import TensorFlow and print the version.
2. Load and normalize the MNIST dataset.
3. Define a neural network with:
   - A flatten layer,
   - A dense layer with ReLU,
   - A dropout layer, and
   - An output layer with 10 units.
4. Generate predictions on a sample and apply softmax to see probabilities.
5. Define and calculate initial loss using Sparse Categorical Crossentropy.
6. Compile the model with the Adam optimizer, loss function, and accuracy metric.
7. Train the model for 5 epochs on the training data.
8. Evaluate the model on test data to check accuracy and loss.

## Code Explanation Step by Step:

### Import TensorFlow Library:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

This imports the TensorFlow library and prints the version to confirm it's installed correctly.

### Load the MNIST Dataset:

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

This loads the MNIST dataset, which consists of images of handwritten digits (0–9). `x_train` and `y_train` contain the training images and labels, while `x_test` and `y_test` are the test images and labels.

### Normalize the Data:

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

This normalizes the pixel values of the images to a range between 0 and 1. Since the original pixel values range from 0 to 255, dividing by 255 scales them to the 0–1 range, making the model training more stable.

### Define the Model Architecture:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

This lesson defines a neural network model with the following layers:
- **Flatten Layer**: Flattens the 28x28 input images into a 1D array of 784 pixels.
- **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation, which introduces non-linearity.
- **Dropout Layer**: Randomly drops 20% of the neurons during training, which helps prevent overfitting.
- **Output Dense Layer**: A fully connected layer with 10 neurons (one for each digit class).

### Generate Predictions:

```python
predictions = model(x_train[:1]).numpy()
```

This runs a forward pass using a single training sample (`x_train[:1]`) to generate predictions before the model is trained. The output is stored in `predictions`.

### Apply Softmax to the Predictions:

```python
tf.nn.softmax(predictions).numpy()
```

This applies the softmax function to `predictions`, converting the raw scores into probabilities for each class (0–9).

### Define the Loss Function:

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

This sets the loss function to Sparse Categorical Crossentropy, which is commonly used for classification tasks with integer labels. Setting `from_logits=True` indicates that the output layer has not applied softmax.

### Calculate Initial Loss:

```python
loss_fn(y_train[:1], predictions).numpy()
```

This calculates the loss for the first sample before training the model, providing an initial loss value.

### Compile the Model:

```python
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
```

Compiles the model, specifying:
- Adam optimizer, which is an efficient optimization algorithm.
- The previously defined `loss_fn`.
- Accuracy as a metric to monitor during training and evaluation.

### Train the Model:

```python
model.fit(x_train, y_train, epochs=5)
```

Trains the model on the training data (`x_train`, `y_train`) for 5 epochs (full passes over the training dataset).

### Evaluate the Model:

```python
model.evaluate(x_test, y_test, verbose=2)
```

Evaluates the model's performance on the test dataset, reporting the accuracy and loss on data it hasn't seen before.
```

Similar code found with 3 license types
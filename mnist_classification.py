# Import necessary libraries
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

dataset = mnist.load_data('mnist.db')

train,test = dataset

len(train)

X_train, y_train = train

X_test, y_test = test

len(X_train)

len(X_test)

X_train[0].shape

X_train[0].ndim

img = X_train[1]

X_train = X_train.reshape(-1,28*28)

X_train.shape

X_test = X_test.reshape(-1, 28*28)

X_test.shape

y_train

y_train = to_categorical(y_train)

y_train

model=Sequential()

model.add( 
    Dense(units = 512 , input_shape = (784,) , activation = 'relu' )
)
model.add( 
    Dense(units = 256 , activation = 'relu' )
)
model.add( 
    Dense(units = 128 , activation = 'relu' )
)
model.add( 
    Dense(units = 64 , activation = 'relu' )
)
model.add( 
    Dense(units = 10 , activation = 'softmax' )
)

model.summary()

model.compile(
    optimizer='adam',
    
    loss='categorical_crossentropy',
    
    metrics=['accuracy']
)

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, to_categorical(y_test))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()


# Predict with the Model:
# Select multiple sample images from the test set (e.g., the first 10 images)
sample_images = X_test


# Use the trained model to make predictions on the selected images
predictions = model.predict(sample_images)


# Print the predictions
print(predictions)


# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)


# Print the predicted labels
print("Predicted Labels:", predicted_labels)


# Generate a confusion matrix and classification report
y_true = y_test
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
confusion = confusion_matrix(y_true, y_pred_labels)
classification_report_str = classification_report(y_true, y_pred_labels)

print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report_str)


# Use the trained model to make predictions on the test set
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Select some random test samples (e.g., the first 15)
num_samples_to_display = 15
sample_indices = np.random.choice(len(X_test), num_samples_to_display, replace=False)

# Create a figure to display the comparison
plt.figure(figsize=(12, 6))

for i, sample_index in enumerate(sample_indices):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[sample_index].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {y_test[sample_index]}\nPredicted: {predicted_labels[sample_index]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
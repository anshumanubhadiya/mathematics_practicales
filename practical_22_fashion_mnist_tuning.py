"""Practical 22: Tune the learning rate of a neural network on the Fashion-MNIST dataset."""
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# Data load aur normalize karna
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = to_categorical(y_train)

# MLP Model Architecture define karna
def build_model():
    return Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

# Learning Rates ke saath Hyperparameter Sweep
for lr in [0.1, 0.01, 0.001]:
    model = build_model()
    model.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=0)
    print(f"lr={lr:.4f}, Validation Accuracy={history.history['val_accuracy'][-1]:.4f}")
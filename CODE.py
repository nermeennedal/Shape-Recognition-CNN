import os
import numpy as np
from PIL import Image


class CNN:
    def __init__(self):
        # Kernels
        self.kernels = [
            np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])/9,  # horizontal line
            np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])/9,  #  vertical line
           np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])/9,       #  plus sign
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])/9,        #  x (\ diagonal)
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])/9,        #  x (/ diagonal)
           np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/9 ,#   square
           np.array([[0, -1, 0], [-1, 1, -1], [1, 1, 1]])/9,     #  triangle
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])/9,       #  circle
            np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])/9,    #  hexagon
            np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])/9,    #  pentagon
            np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])/9       #  diamond
            ]
       
    
        self.num_kernels = len(self.kernels)
        self.nn_input_size = 8 * 8 * self.num_kernels
        self.hidden_layer_size = 64
        self.output_size = 10

        # Weights
        self.W1 = np.random.randn(self.nn_input_size, self.hidden_layer_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_layer_size))
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        

        
        

    def convolve(self, image, kernel):
        h, w = image.shape
        kh, kw = kernel.shape
        output = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
        return output

    def max_pool(self, feature_map, pool_size=2):
        """Apply max pooling to reduce spatial dimensions."""
        height, width = feature_map.shape
        pooled_height = height // pool_size
        pooled_width = width // pool_size

        pooled_map = np.zeros((pooled_height, pooled_width))

        for i in range(pooled_height):
            for j in range(pooled_width):
                start_i = i * pool_size
                start_j = j * pool_size
                pooled_map[i, j] = np.max(feature_map[start_i:start_i + pool_size, start_j:start_j + pool_size])

        return pooled_map

    def relu(self, x): 
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, image):
        self.cache = {}
        conv_outputs = [np.maximum(0, self.convolve(image, k)) for k in self.kernels]
        self.cache['conv_outputs'] = conv_outputs

        pooled_outputs = [self.max_pool(c) for c in conv_outputs]
        self.cache['pooled_outputs'] = pooled_outputs

        flattened = np.concatenate([p.flatten() for p in pooled_outputs])
        self.cache['flattened'] = flattened

        z1 = flattened.dot(self.W1) + self.b1
        a1 = self.relu(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1

        z2 = a1.dot(self.W2) + self.b2
        a2 = self.softmax(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        return a2

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        a2 = self.cache['a2']
        a1 = self.cache['a1']
        flattened = self.cache['flattened']

        dz2 = a2.copy()
        dz2[range(m), y] -= 1
        dz2 /= m
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = dz2.dot(self.W2.T) * (a1 > 0)
        dW1 = flattened.reshape(m, -1).T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


    def predict(self, image):
        probs = self.forward(image)
        return np.argmax(probs)

    def train(self, X_train, y_train, epochs=30, learning_rate=0.005, batch_size=32):
        m = X_train.shape[0]
        losses, accuracies = [], []
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_train = X_train[indices]
            y_train = y_train[indices]

            batch_losses, batch_accuracies = [], []
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                for j in range(len(X_batch)):
                    pred = self.forward(X_batch[j])
                    loss = -np.log(pred[0, y_batch[j]] + 1e-8)
                    batch_losses.append(loss)
                    acc = int(np.argmax(pred) == y_batch[j])
                    batch_accuracies.append(acc)
                    self.backward(X_batch[j].reshape(1, -1), np.array([y_batch[j]]), learning_rate)

            losses.append(np.mean(batch_losses))
            accuracies.append(np.mean(batch_accuracies))
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

        return losses, accuracies


def load_and_preprocess_data(data_dir):
    images, labels = [], []
    for sign_number in range(10):  
        for i in range(1, 1000):  
            img_name = f"{sign_number}_{i}.PNG"
            img_path = os.path.join(data_dir, img_name)
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert('L').resize((19, 19))
                img_array = np.array(img) / 255.0
                img_array = (img_array < 0.5).astype(np.float32)
                images.append(img_array)
                labels.append(sign_number)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)


def train_cnn(output_file="cnn_weights_team4.npz", epochs=30, learning_rate=0.005):
    # Load and preprocess data
    data_dir = "C:/Users/nerme/Desktop/trainfinal"
    print(f"Loading and preprocessing data from {data_dir}")
    images, labels = load_and_preprocess_data(data_dir)
    
    if len(images) == 0:
        print("No images found. Check the data directory path.")
        return None, None, None
    
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} classes")
    
    # Create and train CNN
    print(f"Creating and training CNN for {epochs} epochs")
    cnn = CNN()
    losses, accuracies = cnn.train(images, labels, epochs=epochs, learning_rate=learning_rate)
    
    # Save the trained model weights
    print(f"Saving model weights to {output_file}")
    np.savez(output_file, W1=cnn.W1, b1=cnn.b1, W2=cnn.W2, b2=cnn.b2)
    
    # Final training accuracy
    final_accuracy = accuracies[-1] if accuracies else 0
    print(f"Final training accuracy: {final_accuracy:.4f}")
    
    return cnn, losses, accuracies

def test_cnn_on_directory(model_path, test_data_dir):
    print("Loading trained model weights")
    cnn = CNN()
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    data = np.load(model_path)
    cnn.W1 = data["W1"]
    cnn.b1 = data["b1"]
    cnn.W2 = data["W2"]
    cnn.b2 = data["b2"]

    print("Loading and preprocessing test data")
    test_images, test_labels = load_and_preprocess_data(test_data_dir)
    if len(test_images) == 0:
        print("No valid images found in test directory")
        return

    print("Running predictions")
    correct = 0
    for i in range(len(test_images)):
        pred = cnn.predict(test_images[i])
        true = test_labels[i]
        print(f"Image {i+1}: Predicted = {pred}, Actual = {true}")
        if pred == true:
            correct += 1

    accuracy = correct / len(test_images)
    accuracy = correct / len(test_images)
    incorrectly_predicted_images=len(test_images)-correct
    print(f"Test Accuracy on {len(test_images)} samples: {accuracy:.4f}")
    print(f"Number of correctly predicted images : {correct} out of {len(test_images)}")
    print (f"Number of incorrectly predicted images : {incorrectly_predicted_images} out of {len(test_images)}")


#train
#train_cnn()
#test
#test_dir = "C:/Users/nerme/Desktop/TestingImages/TestingImages/TestingImgs4"
current_directory = os.getcwd()
images_directory = current_directory+'/../GRADING_IMAGES/'
test_cnn_on_directory("cnn_weights_team4.npz", images_directory)


#  cnn shape classifier

this project implements a simple cnn from scratch using numpy to classify 19x19 binary images of geometric shapes.

##  dataset
- 10 shape classes: horizontal line, vertical line, plus, x, square, triangle, circle, hexagon, pentagon, diamond
- grayscale png images (19x19), binarized to black and white
- training images must be named as `{label}_{version}.PNG` inside the training folder

## 🏗 model architecture
- 11 manually designed 3x3 convolution filters
- relu activation
- 2x2 max pooling
- 1 fully connected hidden layer (64 units)
- softmax output layer

##  testing
```python
# inside your script, test the model using:
test_cnn_on_directory("cnn_weights_team4.npz", images_directory)

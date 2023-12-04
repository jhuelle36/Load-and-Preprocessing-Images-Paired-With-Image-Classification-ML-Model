# Load-and-Preprocessing-Images-Paired-With-Image-Classification-ML-Model
This file uses three different methods of loading and preprocessing images using TensorFlow. Paired with this loading and preprocessing practice is a 10-layer Convolutional Neural Network designed for image classification. This model loads thousands of photos of flowers which are separated for training, validating, and testing the model. The goal is to classify the type of flower that the picture contains. I learned how to create this model using TensorFlow tutorials.

CNN Model Layer Specifications: 
- Rescaling Layer: For preprocessing images
- 2D Convolutional layer 1: 32 neuron layer used for detecting simple patterns like edges
- MaxPooling Layer 1: Pooling layer that reduces spatial dimensions
- 2D Convolutional layer 2: 32 neuron layer used for detecting more complex patterns from previous simple patterns like parts of the flower
- MaxPooling layer 2: Pooling layer that reduces spatial dimensions
- 2D Convolutional Layer 3: 32 neuron layer used for detecting complex patterns from previous patterns found in other layers
- MaxPooling layer 3: Pooling layer that reduces spatial dimensions
- Flatten Layer: Flatten's output from previous convolutional layers to a 1D array for dense layers
- Dense Layer: 128 neuron layer for putting together all of the previously learned patterns to find more useful sophisticated patterns
- Output Layer: 5 neuron dense layer for final classification



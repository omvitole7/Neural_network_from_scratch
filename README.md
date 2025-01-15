# Neural Network Implementation from Scratch

## Objective
This project implements a simple feedforward neural network from scratch in Python without using any in-built deep learning libraries. It covers basic components such as:
- Forward Pass
- Backpropagation
- Training using Gradient Descent

## Problem Definition
The neural network is trained to solve a **binary classification problem** based on the **AND logic gate**.

### Dataset
| Input (X)   | Output (Y) |
|-------------|------------|
| [0, 0]      | [0]        |
| [0, 1]      | [0]        |
| [1, 0]      | [0]        |
| [1, 1]      | [1]        |

### Neural Network Architecture
1. **Input Layer**: 2 neurons representing the binary inputs of the AND operation.
2. **Hidden Layer**: 3 neurons with Sigmoid activation function.
3. **Output Layer**: 1 neuron with Sigmoid activation function.

### Forward Pass
1. Input data is passed through the input layer.
2. The weighted sum is calculated and passed through the hidden layer with the activation function applied.
3. The hidden layer output is further processed through the output layer to generate the final prediction.

### Backpropagation
1. Errors are computed as the difference between predicted and actual output values.
2. Gradients are calculated and used to update weights and biases using Gradient Descent.

### Loss Function
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.

---

## Methodology
1. **Sigmoid Activation Function**:
   ```python
   def sigmoid(x):
       return 1 / (1 + np.exp(-x))
   ```

2. **Sigmoid Derivative**:
   ```python
   def sigmoid_derivative(x):
       return x * (1 - x)
   ```

3. **Neural Network Class**:
   A class is created to define the forward pass, backpropagation, and training process. The network uses randomly initialized weights and biases, updated iteratively.

4. **Training**:
   The network is trained for 10,000 epochs using a learning rate of 0.05.

---

## Results
After training, the network predicts the output for the AND logic gate with high accuracy.

### Example Output
| Input (X)   | Predicted Output |
|-------------|------------------|
| [0, 0]      | ~0.01           |
| [0, 1]      | ~0.03           |
| [1, 0]      | ~0.03           |
| [1, 1]      | ~0.94           |

### Loss During Training
- **Epoch 0**: 0.2400
- **Epoch 1000**: 0.1001
- **Epoch 5000**: 0.0047
- **Epoch 9000**: 0.0018

---

## Usage

### Prerequisites
- Python 3.x
- NumPy

### Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/omvitole7/Neural_network_from_scratch.git
   cd Neural_network_from_scratch
   ```
2. Execute the script:
   ```bash
   python neural_network.py
   ```

### Customize the Training

- **Number of Hidden Neurons**
- **Learning Rate**
- **Number of Epochs**

---

## Project Details

- **Author**: Om Vitole  



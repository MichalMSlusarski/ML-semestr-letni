# Notes and projects from my Machine Learning classes

While machine learning is not one of the easiest topics, the basic principle of how neural networks operate can be summed up in one sentence:
*We aim to find the **minimum of the loss function**, meaning the **weights**' values for which the cost function takes the smallest value.*

### Repository Contents

- Network from scratch
- Introduction to Keras (not yet translated)
- Word Embeddings (not yet translated)

## Network from scratch

Although the contemporary machine learning landscape is dominated by high-level libraries such as TensorFlow or PyTorch, at the very beginning, it's worth experimenting with networks written from scratch. Implementing a simple neural network from scratch allows one to understand fundamental concepts such as backpropagation, activation functions, synaptic weighting, etc.

Implementations of simple networks can be found in the folder: ![Neural-Network-From-Scratch](https://github.com/MichalMSlusarski/ML-semestr-letni/tree/main/Neural-Network-From-Scratch)

Description of individual stages of network creation, experiments, and defined functions.

### Data Preparation

For training and modifying the network, we received a test set named basic.csv containing basic metrics of posts on Facebook. Our task is to check, clean, divide the dataset, and define the predicted value. **It's worth noting right away that the set is very small - too small for a normal network to yield good results.**

I start working with the data by visually inspecting the dataset. I determined that there are two rows explaining the name and type of the presented variable. There are also numerous empty values, most likely representing missing values.

To clean the dataset, I created a script called clean_data. As part of the operation, I load the uncleaned set basic.csv. I perform basic cleaning: removing explanatory rows and the Z13 column; filling in missing data with zeros and proactively standardizing the data type in the table (to_numeric).

```python
import pandas as pd
import numpy as np

df = pd.read_csv('basic.csv')

df = df.drop([0, 1])
df = df.reset_index(drop=True)
df = df.drop('Z13', axis=1)
df = df.fillna(0)
df = df.apply(pd.to_numeric)
```
The next step is standardizing the data so that the values fall within a similar range. Without standardization, high values could dominate the network learning process.

For standardization, I choose numerical columns (I consider all others as categorical: 0-1). I use the max-min method, which scales values in specified columns to a standard range from 0 to 1. This is done by subtracting the minimum value in the columns and dividing it by the range of values found there.
```python
columns_to_standardize = ['Z11', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']
df[columns_to_standardize] = round((df[columns_to_standardize] - df[columns_to_standardize].min()) / (df[columns_to_standardize].max() - df[columns_to_standardize].min()), 6)
```
I convert variable Z21 from numerical to categorical, considering the median of the number of comments as the threshold. This allows the dataset to be divided into two roughly equal classes.
```python
median = df['Z21'].median()
df['Z21'] = np.where(df['Z21'] > median, 1, 0)
```
Next, I check if the obtained classes are indeed approximately equal.
```python
print(df['Z21'].value_counts())
```
For the median, the class counts are as follows:
0    341
1    301

By comparison, when grouped by mean, the classes were significantly unequal:
0    488
1    154

The median value is 7. This means that 341 posts have exactly 7 or fewer comments, while 301 posts have more. The classes cannot be exactly equal as the values would overlap. The task of the network will be to predict whether, for a given post, the number of comments is greater than the dataset's median.

Towards the end of the processing, I allocate the last 50 entries from the dataset for the test set, removing them from the training set.
```python
test_df = df[-50:]
df = df[:-50]
```
I save the datasets.
```python
df.to_csv('training.csv')
test_df.to_csv('test.csv')
```

### Building the Neural Network

Initially, I tested an architecture based on code from classes.
```python
import pandas as pd
import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
```

The constructor __init__(self) initializes the synaptic weights (synaptic_weights) with random values from the range [-1, 1].

The sigmoid activation function sigmoid(self, x) calculates the value of the sigmoid function for a given value x.

**Sigmoid function formula**

$\sigma(z) = \frac{1} {1 + e^{-z}}$

where $z$ is the input vector - the weighted sum of neuron inputs. This function returns a value close to 0 for very large negative values of $x$ and close to 1 for very large positive values of $x$. For $x$ equal to zero, the sigmoid function returns a value of 0.5.

The derivative of the sigmoid function sigmoid_derivative(self, x) calculates the derivative of the sigmoid function for a given value x. This derivative is used when calculating weight adjustments in the training process using backpropagation.

The training method train(self, training_inputs, training_outputs, training_iterations) goes through a specified number of training iterations. For each iteration:

- It performs a "forward pass" through the network, calculating the output based on training inputs.
- It calculates the error as the difference between expected training outputs and the actual outputs of the network.
- It calculates weight adjustments based on the dot product of the transposed matrix of training inputs, the error, and the derivative of the sigmoid function from the output.
- It updates the synaptic weights by adding the adjustments.

The prediction method think(self, inputs) takes inputs and calculates the output of the neural network. It transforms inputs into float type, performs a weighted sum of inputs (np.dot(inputs, self.synaptic_weights)), and applies the sigmoid function to the result. It returns the calculated output, interpreted as the probability of the positive class in binary classification.

### Model

 Training

What has changed in relation to the class script is the way data is inputted and validated.

In the if declaration ```__name__ == "__main__":```, I start by importing data from the training set.
```python
df = pd.read_csv("training.csv")
```
I extract variables Z9-Z20 as explanatory/predictive variables and Z21 as the dependent variable (labels).
```python
input_features = df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
output_labels = df[['Z21']]
```
Similar to the original script, values from specific columns are assigned as inputs and outputs of the network.
```python
training_inputs = input_features.values
training_outputs = output_labels.values
```
The network object is initialized, and the training method is called.
```python
neural_network = NeuralNetwork()
neural_network.train(training_inputs, training_outputs, 11000)
```
The obtained weights are saved in a file named weights.npy. The .npy extension is part of the NumPy library.
```python
np.save("weights.npy", neural_network.synaptic_weights)
```

### Model Testing

I move on to validate the model. Initially, I initialize the network by loading the saved weights.
```python
neural_network = NeuralNetwork()
neural_network.synaptic_weights = np.load("weights.npy")
```
Similar to the training set, I prepare the test set.
```python
test_df = pd.read_csv("test.csv")
test_input_features = test_df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
test_output_labels = test_df[['Z21']]
test_inputs = test_input_features.values
test_outputs = test_output_labels.values
```
Test values are fed into the model and assigned to the variable predictions. Since these are floating-point values, I convert them to binary by calling the astype() method. Values that exceed the threshold of 0.5 are counted as 1, while the rest are counted as 0.
```python
predictions = neural_network.think(test_inputs)
binary_predictions = (predictions > 0.5).astype(int)
```
The metric I'll use to check the model is accuracy. I calculate the accuracy of the model by averaging the cases where the prediction matches the actual value. Then, the result is multiplied by 100 to obtain the value in percentage.
For example, if accuracy is 85, it means the model correctly classified 85% of the samples from the test set.
```python
accuracy = np.mean(binary_predictions == test_outputs) * 100
print("Mean accuracy:", accuracy)
```
The average accuracy of the model was 70%. It's not an outstanding result, but it's quite decent considering a single layer and a **very small training set**.
```

Certainly, I'll translate the provided Polish Markdown section into English while preserving its structure and code formatting:

```markdown
### Model Modification

#### Adding an Intermediate Layer

The first modification to the architecture that could potentially improve its accuracy is adding an intermediate layer, also known as a hidden layer. Adding such a layer begins with declaring a new set of weights for each of the layers.

The input layer has 11 neurons accepting the training input. These 11 neurons are connected to 5 neurons in the hidden layer, which, in turn, connects to a single output neuron. The choice of 5 neurons resulted from further experiments.
```python
def __init__(self):
    np.random.seed(1)
    self.synaptic_weights1 = 2 * np.random.random((11, 5)) - 1
    self.synaptic_weights2 = 2 * np.random.random((5, 1)) - 1
```
The modification occurs in the training function. I'm adding the output of the hidden layer. Its declaration is similar to the output of the entire function in the single-layer version. However, in the single-layer version, it only appears in the think function. Here, it needs to be synchronized with the hidden layer in each iteration. It's crucial to remember that the output of the entire function now takes the hidden output, not the training one.
```python
def train(self, training_inputs, training_outputs, training_iterations):
    for iteration in range(training_iterations):
        hidden_output = self.sigmoid(np.dot(training_inputs, self.synaptic_weights1))
        output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
```
Similar to before, the error in the output layer is calculated based on the difference between expected and predicted outputs. Then adjustments (output_adjustments) are calculated using the derivative of the activation function of the output layer, just as in the case of a single-layer network.

Based on these adjustments, the error in the hidden layer is calculated using the weight matrix between the hidden and output layers. By multiplying these corrections by the synaptic_weights2, the error from the output layer is passed back to the hidden layer. This step allows attributing part of the error in the output layer to each neuron in the hidden layer.
```python
output_error = training_outputs - output
output_adjustments = output_error * self.sigmoid_derivative(output)
hidden_error = np.dot(output_adjustments, self.synaptic_weights2.T)
hidden_adjustments = hidden_error * self.sigmoid_derivative(hidden_output)
```
The hidden output is declared similarly to the output:
```python
def think(self, inputs):
    inputs = inputs.astype(float)
    hidden_output = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
    output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
    return output
```
Testing the model follows the same principle as before. However, it turns out that the model loses accuracy. After several experiments with the number of intermediate neurons, the best results were achieved with 5 neurons. Nevertheless, these results are worse than in the single-layer network, reaching only 54%.

Most likely, the additional layer introduces unnecessary complexity to a simple relationship observed in the data. Moreover, multi-layer models may require a larger amount of training data. Alternatively, trying to define a different, simpler activation function in the hidden layer might be worth considering.

#### Changing the Activation Function

A simple activation function that can be applied in the hidden layer is ReLU. It assigns the value of positive inputs linearly and sets negative inputs to 0.

**ReLU Function Formula**

$Relu(z) = max(0, z)$

I utilize the maximum function from the NumPy library to define the ReLU function.
```python
def relu(self, x):
    return np.maximum(0, x)
```
The derivative of ReLU is the derivative of X for positive numbers and 0 for negative numbers. Hence, its values are 1 and 0, respectively.
```python
def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)
```

The only change compared to the two-layer network is the necessity of replacing the sigmoid function with ReLU wherever it appeared as a hidden layer. An example is given below:
```python
hidden_output = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
```
Which I change to:
```python
hidden_output = self.relu(np.dot(training_inputs, self.synaptic_weights1))
```
Experimenting with the number of intermediate neurons, with 7 neurons, the model achieved a maximum accuracy of 64%. Better than in the case of two layers with sigmoid activation but still worse than in the case of a single-layer network. As seen, simpler doesn't mean worse.
```


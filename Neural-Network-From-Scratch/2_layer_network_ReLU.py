import pandas as pd
import numpy as np

# *******************************
# * DWUWARSTWOWA SIEĆ NEURONOWA *
# *  z funkcją aktywacji ReLU   *                                                                   
# *******************************

# Komentarzami oznaczyłem miejsca, w których coś zmieniłem w stosunku do sieci dwuwarstwowej - 2_layer_network

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        # w przypadku relu 7 neuronów radzi sobie lepiej niż 5
        self.synaptic_weights1 = 2 * np.random.random((11, 7)) - 1
        self.synaptic_weights2 = 2 * np.random.random((7, 1)) - 1

    # wykorzystuję funkcję maximum do zdefiniowania ReLU
    def relu(self, x):
        return np.maximum(0, x)
    
    # pochodna z ReLU to po prostu pochodna z X, czyli 1 (albo zero dla ujemnych)
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            # tutaj modyfikacje są identyczne jak w przypadku dwóch warstw sigmoidalnych, zmianie ulega tylko pierwsza funkcja
            hidden_output = self.relu(np.dot(training_inputs, self.synaptic_weights1))
            output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
            output_error = training_outputs - output
            output_adjustments = output_error * self.sigmoid_derivative(output)

            hidden_error = np.dot(output_adjustments, self.synaptic_weights2.T)
            hidden_adjustments = hidden_error * self.relu_derivative(hidden_output)

            self.synaptic_weights2 += np.dot(hidden_output.T, output_adjustments)
            self.synaptic_weights1 += np.dot(training_inputs.T, hidden_adjustments)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        inputs = inputs.astype(float)
        # tutaj też podmieniam sigmoid na relu
        hidden_output = self.relu(np.dot(inputs, self.synaptic_weights1))
        output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))
        return output



if __name__ == "__main__":

    # ******************
    # * TRENING MODELU *                                                                          
    # ******************

    df = pd.read_csv("training.csv")

    input_features = df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
    output_labels = df[['Z21']]

    training_inputs = input_features.values
    training_outputs = output_labels.values

    neural_network = NeuralNetwork()
    neural_network.train(training_inputs, training_outputs, 11000)

    # inna ścieżka:
    np.save("weights_2_layers_ReLU.npy", neural_network.synaptic_weights2)

    # *********************
    # * TESTOWANIE MODELU *                                                                          
    # *********************

    neural_network = NeuralNetwork()
    neural_network.synaptic_weights2 = np.load("weights_2_layers_ReLU.npy") # tu też dostosowuję ścieżkę

    test_df = pd.read_csv("test.csv")
    test_input_features = test_df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
    test_output_labels = test_df[['Z21']]
    test_inputs = test_input_features.values
    test_outputs = test_output_labels.values

    predictions = neural_network.think(test_inputs)
    binary_predictions = (predictions > 0.5).astype(int)

    accuracy = np.mean(binary_predictions == test_outputs) * 100
    print("Mean accuracy:", accuracy)
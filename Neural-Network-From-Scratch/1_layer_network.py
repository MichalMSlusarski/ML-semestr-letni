import pandas as pd
import numpy as np

# *********************************
# * JEDNOWARSTWOWA SIEĆ NEURONOWA *                                                                          
# *********************************

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

if __name__ == "__main__":

    # ******************
    # * TRENING MODELU *                                                                          
    # ******************

    # ładuję set treningowy i wyciągam zmienne Z9-Z20 jako zmienne objaśniające/predykcyjne i Z21 jako objaśnianą (target)
    df = pd.read_csv("training.csv")

    input_features = df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
    output_labels = df[['Z21']]

    training_inputs = input_features.values
    training_outputs = output_labels.values

    neural_network = NeuralNetwork()
    neural_network.train(training_inputs, training_outputs, 11000)

    # zapisuję uzyskane wagi
    np.save("weights.npy", neural_network.synaptic_weights)

    # *********************
    # * TESTOWANIE MODELU *                                                                          
    # *********************

    # inicjuję siec i wczytuję zapisane wagi
    neural_network = NeuralNetwork()
    neural_network.synaptic_weights = np.load("weights.npy")

    # analogicznie do treningu przygotowuję zbiór testowy
    test_df = pd.read_csv("test.csv")
    test_input_features = test_df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
    test_output_labels = test_df[['Z21']]
    test_inputs = test_input_features.values
    test_outputs = test_output_labels.values

    # sprawdzam model
    predictions = neural_network.think(test_inputs)
    binary_predictions = (predictions > 0.5).astype(int)

    # obliczam dokładność modelu wyciągajc średnią z przypadków gdzie predykcja jest równa rzeczywistości
    accuracy = np.mean(binary_predictions == test_outputs) * 100
    print("Mean accuracy:", accuracy)
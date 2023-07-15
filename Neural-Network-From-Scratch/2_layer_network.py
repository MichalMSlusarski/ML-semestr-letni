import pandas as pd
import numpy as np

# *******************************
# * DWUWARSTWOWA SIEĆ NEURONOWA *                                                                          
# *******************************

# Komentarzami oznaczyłem miejsca, w których coś zmieniłem w stosunku do sieci jednowarstwowej - 1_layer_network

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        # dodaję nową warstwę, eksperymentując z liczbą neuronów pośrednich (tutaj 5)
        self.synaptic_weights1 = 2 * np.random.random((11, 5)) - 1
        self.synaptic_weights2 = 2 * np.random.random((5, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):
            # definiuję output warstwy ukrytej, analogicznie do tego jak wcześniej był definiowany output ostateczny
            hidden_output = self.sigmoid(np.dot(training_inputs, self.synaptic_weights1))

            # w dwuwarstwowej sieci trzeba dostosować output w każdej iteracji, więc deklarowany jest też tu:
            output = self.sigmoid(np.dot(hidden_output, self.synaptic_weights2))

            output_error = training_outputs - output
            output_adjustments = output_error * self.sigmoid_derivative(output)

            # błąd w warstwie ukrytej opisałem dokładniej w raporcie
            hidden_error = np.dot(output_adjustments, self.synaptic_weights2.T)
            hidden_adjustments = hidden_error * self.sigmoid_derivative(hidden_output)

            self.synaptic_weights2 += np.dot(hidden_output.T, output_adjustments)
            self.synaptic_weights1 += np.dot(training_inputs.T, hidden_adjustments)

    def think(self, inputs):
        inputs = inputs.astype(float)
        # warstwa ukryta deklarowana jest w tak samo jak wcześniej
        hidden_output = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
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

    # Uaktualniam ścieżkę
    np.save("weights_2_layers.npy", neural_network.synaptic_weights2)

    # *********************
    # * TESTOWANIE MODELU *                                                                          
    # *********************

    neural_network = NeuralNetwork()
    neural_network.synaptic_weights2 = np.load("weights_2_layers.npy")

    test_df = pd.read_csv("test.csv")
    test_input_features = test_df[['Z9', 'Z10','Z11', 'Z12', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20']]
    test_output_labels = test_df[['Z21']]
    test_inputs = test_input_features.values
    test_outputs = test_output_labels.values

    predictions = neural_network.think(test_inputs)
    binary_predictions = (predictions > 0.5).astype(int)

    accuracy = np.mean(binary_predictions == test_outputs) * 100
    print("Mean accuracy:", accuracy)
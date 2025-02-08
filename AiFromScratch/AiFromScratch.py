import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
X = X/255.0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

print("Training dataset Loaded!")

class Functions:
    @staticmethod
    def relu(x):
        if x <= 0:
            return 0
        else:
            return x

    @staticmethod
    def compute_forward_matrix(weights, inputs, biases, relu=True):
        forward_matrix = np.dot(weights, inputs) + biases
        if relu:
            forward_matrix = np.maximum(0, forward_matrix)

        return forward_matrix
       
    @staticmethod
    def get_random_weights(inputs: int, outputs: int):  # uses He Initialization, setting bias to 0
        std = np.sqrt(2/inputs)
        random_weights = np.random.normal(0, std, size=(outputs, inputs))
        return random_weights

    @staticmethod
    def softmax(x: np.array):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class Layer:
    def __init__(self, input_size, output_size, batch_size=8):
        self.inputs = input_size
        self.outputs = output_size
        self.biases = np.zeros(output_size)
        self.weights = np.zeros((output_size, input_size))
        self.activation = True

        self.input_history = [] # Inputs to the neurons, before weights and bias
        self.z_history = [] # Pre Activation function
        self.output_history = [] # outputs, after ReLU or similar activation function
        self.truth_history = [] 

        self.weight_gradients = None
        self.bias_gradients = None

        self.batch_size = batch_size

    def initialize_layer(self):
        self.weights = Functions.get_random_weights(self.inputs, self.outputs)

    def forward(self, input_vector, compute_grads=True):
        if compute_grads:
            z = np.dot(self.weights, input_vector) + self.biases
            output = np.maximum(0, z) if self.activation else z
            self.input_history.append(input_vector)
            self.z_history.append(z)
            self.output_history.append(output)

        else:
            if self.activation:
                relu=True
            else:
                relu = False

            output = Functions.compute_forward_matrix(self.weights, input_vector, self.biases, relu)

        return output

    def clear_batch_memory(self):
        self.input_history = []
        self.z_history = []
        self.output_history = []

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_biases(self, new_biases):
        self.biases = new_biases

class NeuralNetwork:
    def __init__(self, batch_size):
        self.layers = []
        self.batch_size = batch_size

    def forward(self, x):
        for layer_obj in self.layers:
            x = layer_obj.forward(x)
        return x

    def build_fcl(self, input_size, output_size):
        new_layer = Layer(input_size, output_size, self.batch_size)
        new_layer.initialize_layer()
        self.layers.append(new_layer)

    def get_layer_weights(self, layer):
        if abs(layer) > len(self.layers):
            raise ValueError("Layer number requested exceeds total number of network layers")
        return self.layers[layer].weights

    def get_layer(self, layer) -> Layer:
        if abs(layer) > len(self.layers):
            raise ValueError("Layer number requested exceeds total number of network layers")
        return self.layers[layer]

    def zero_grad(self):
        for layer in self.layers:
            layer.clear_batch_memory()

class Trainer:
    def __init__(self, NN, batch_size=1):
        self.nn = NN  # object of NeuralNetwork()
        self.number_layers = len(NN.layers)
        self.batch_size = batch_size
        self.gradients_list = []
        self.bias_gradients_list = []

    def run_batch(self, training_batch: list[np.array], training_answers):  # run the batch through the neural network, returns the avg loss (CEL)
        one_hot_encoding = [[1 if x == label else 0 for x in range(10)] for label in training_answers]
        losses = []
        for idx, image in enumerate(training_batch):
            output = self.nn.forward(image)
            losses.append(self.CrossEntropyLoss(one_hot_encoding[idx], output))
            self.nn.get_layer(-1).truth_history.append(one_hot_encoding[idx])

        avg_loss = np.mean(losses)
        return avg_loss

    def CrossEntropyLoss(self, truth_vector: np.array, output_vector: np.array):
        #  Truth Vector is the answer vector ([0, 0, 1, 0, 0])
        # output Vector is what the model predicts ([0.12, 0.10, 0.98, 0.4, 0.1])
        loss = 0
        for category_index, category_result in enumerate(truth_vector):
            loss -= category_result * np.log(output_vector[category_index] + 1e-8)
        
        return loss

    def MSELoss(self, truth_vector: np.array, output_vector: np.array):
        diff = truth_vector - output_vector
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)
        return mse

    def compute_last_layer_gradients(self, truth_vector, output_vector):
        gradient = (Functions.softmax(output_vector) - truth_vector)
        return gradient

    def compute_gradient(self, upstream_gradient, layer_weights, layer_inputs, layer_outputs,  activation=True):
        if activation:
            relu_gradient = (layer_outputs > 0).astype(float)

        else:
            relu_gradient = layer_outputs
        activation_gradient = upstream_gradient * relu_gradient
        input_gradients = np.dot(activation_gradient, layer_weights.T)
        weight_gradients = np.outer(layer_inputs, activation_gradient)
        return input_gradients, weight_gradients

    def compute_gradients(self):
        self.gradients_list = []
        for layer_from_back, layer in enumerate(reversed(self.nn.layers)):
            layer_gradients = []
            layer_upstream_gradients = []
            for idx_in_batch in range(self.batch_size):
                if layer_from_back == 0:
                    layer_gradient = self.compute_last_layer_gradients(layer.truth_history[idx_in_batch], 
                                                                       layer.output_history[idx_in_batch])
                    upstream_gradient = layer_gradient
                else:
                    layer_weights = layer.weights
                    layer_inputs = layer.input_history[idx_in_batch]
                    layer_outputs = layer.output_history[idx_in_batch]
                    upstream_gradient, layer_gradient = self.compute_gradient(previous_upstream_gradients[idx_in_batch], 
                                                           layer_weights, 
                                                           layer_inputs, 
                                                           layer_outputs, 
                                                           True)
                layer_gradients.append(layer_gradient)
                layer_upstream_gradients.append(upstream_gradient)


            previous_upstream_gradients = layer_upstream_gradients
            all_gradients = np.array(layer_gradients)
            total_gradient = np.mean(all_gradients, axis=0)
            self.gradients_list.append(total_gradient)

    def compute_last_layer_bias_gradients(self, truth_vector, output_vector):
        return Functions.softmax(output_vector) - truth_vector

    def compute_bias_gradient(self, upstream_gradient, layer_outputs, activation=True):
        if activation:
            relu_deriv = (layer_outputs > 0).astype(float)
        else:
            relu_deriv = layer_outputs
        activation_gradient = upstream_gradient * relu_deriv
        return activation_gradient

    def compute_bias_gradients(self):
        self.bias_gradients_list = []
        previous_bias_gradients = None

        for layer_index, layer in enumerate(reversed(self.nn.layers)):
            batch_bias_gradients = []
            batch_upstream_gradients = []
            for idx in range(self.batch_size):
                if layer_index == 0:
                    bias_grad = self.compute_last_layer_bias_gradients(
                        layer.truth_history[idx],
                        layer.output_history[idx]
                    )

                    new_upstream = bias_grad
                else:
                    upstream_bias_gradient = previous_bias_gradients
                    bias_grad = self.compute_bias_gradient(
                            upstream_bias_gradient, layer.output_history[idx], activation=True)
                    new_upstream = np.dot(bias_grad, layer.weights.T)

                batch_bias_gradients.append(bias_grad)
                batch_upstream_gradients.append(new_upstream)

            previous_bias_gradients = batch_upstream_gradients
            avg_bias_grad = np.mean(np.array(batch_bias_gradients), axis=0)
            self.bias_gradients_list.append(avg_bias_grad)

    def compute_all_gradients(self):
        self.compute_gradients()
        self.compute_bias_gradients()

    def zero_grad(self):
        self.nn.zero_grad()
        self.gradients_list = []
        self.bias_gradients_list = []

    def update_weights(self, learning_rate, momentum=0):
        for layer_idx, gradients in enumerate(reversed(self.gradients_list)):
            weights = self.nn.layers[layer_idx].weights
            new_weights = weights - learning_rate * gradients
            self.nn.layers[layer_idx].update_weights(new_weights)

        for layer_idx, bias_gradients in enumerate(reversed(self.bias_gradients_list)):
            biases = self.nn.layers[layer_idx].biases
            new_biases = biases - learning_rate * bias_gradients
            self.nn.layers[layer_idx].update_biases(new_biases)

def print_number(number_array):
    for row in range(28):
        line = ''
        for col in range(28):
            if number_array[row * 28 + col] > 0.5:
                line += 'X'
            else:
                line += ' '
        print(line)

def main():
    batch_size = 16
    lr = 0.01
    nn = NeuralNetwork(batch_size)
    # Defining settings:
    

    # building a simple Fully Connected Neural Network
    nn.build_fcl(28 * 28, 28 * 28)
    nn.build_fcl(28 * 28, 400)
    nn.build_fcl(400, 240)
    nn.build_fcl(240, 120)
    nn.build_fcl(120, 10)

    trainer = Trainer(nn, batch_size)
    image_num = 0
    training_data = []
    training_answers = []
    for image in X_train:
        if image_num % 16 == 0:
            new_batch = []
            new_batch_answers = []
        new_batch.append(image)
        new_batch_answers.append(y_train[image_num])
        if image_num % 16 == 15:
            training_data.append(new_batch)
            training_answers.append(new_batch_answers)
        image_num += 1

    # Training
    # Training data is X_train which is 60,000 28*28 images of hand written digits
    # Correct answer is y_train
    running_loss = 0.0
    for epoch in range(5):
        for idx, batch in enumerate(training_data):
            trainer.zero_grad()
            loss = trainer.run_batch(batch, training_answers[idx])
            trainer.compute_all_gradients()
            trainer.update_weights(lr)
            running_loss -= loss
            if idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {idx}, Running_loss = {running_loss}")
                running_loss = 0.0


    print_number(training_data[0][0])
    print(nn.forward(training_data[0][0]))
    
if __name__ == '__main__':
    main()
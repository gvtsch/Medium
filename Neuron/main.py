import numpy as np
np.random.seed(42)

def get_dataset():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y

def accuracy_score(y_true, y_pred):
    N = y_true.shape[0]
    accuracy = np.sum(y_true == y_pred) / N
    return accuracy

def step_function(input_signal):
    output_signal = (input_signal > 0.0).astype(np.int_)
    return output_signal

class Neuron:
    def __init__(self, learning_rate, input_dim):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.w = np.random.uniform(-1, 1, size=(self.input_dim, 1))

    def _update_weights(self, x, y, y_pred):
            error = y - y_pred
            delta = error * x
            for delta_i in delta:
                self.w = self.w + self.learning_rate * delta_i.reshape(-1, 1)
            print(f"Weights: {self.w}")

    def train(self, x, y, epochs = 1):
        for epoch in range(1, epochs + 1):
            y_pred = self.predict(x)
            self._update_weights(x, y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            print(f"Epoch {epoch}: Accuracy {accuracy}")

    def predict(self, x):
        input_signal = np.dot(x, self.w)
        output_signal = step_function(input_signal)
        return output_signal

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
    
if __name__=='__main__':
    x, y = get_dataset()
    input_dim = x.shape[1]
    learning_rate = 0.1
    p = Neuron(learning_rate, input_dim)
    p.train(x, y, epochs = 10)
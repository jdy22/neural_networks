import numpy as np
import pickle
import traceback
import random

def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        # A stands for the post-activation output layer

        A = 1 / (1 + np.exp(-x))
        self._cache_current = A
        return A


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """

        A = self._cache_current
        dZ = grad_z * A * (1-A)

        return dZ

class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        # A stands for the post-activation output layer
        A = np.maximum(0.0, x) # returns value unchanged if it is bigger than 0
        self._cache_current = A

        return A

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        A = self._cache_current
        dZ = np.multiply(grad_z, np.int64(A > 0))

        return dZ


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self._W = xavier_init(((n_in, n_out)))
        self._b = np.zeros(n_out)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # store normalised x for backward and update_params
        
        self._cache_current = x #also equal to dzdw

        # vector b will be broadcast to accompany with shape of W here 
        return np.matmul(x, self._W) + self._b 


    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        self._grad_W_current = np.matmul(np.transpose(self._cache_current), grad_z)
        self._grad_b_current = np.matmul(np.ones(len(grad_z)), grad_z) # Changed dimension to batch_size + removed transpose
        dLdx = np.matmul(grad_z, np.transpose(self._W))

        return dLdx

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W = self._W - learning_rate * self._grad_W_current
        self._b = self._b - learning_rate * self._grad_b_current


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        self._layers = []
        for layer_index in range(len(neurons)):
            # Create linear layer
            if layer_index == 0:
                n_in = input_dim
            else:
                n_in = neurons[layer_index-1]
            n_out = neurons[layer_index]
            self._layers.append(LinearLayer(n_in, n_out))
            
            # Create activation function
            if activations[layer_index] == "relu":
                self._layers.append(ReluLayer())
            elif activations[layer_index] == "sigmoid":
                self._layers.append(SigmoidLayer())


    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        output = x
        for layer in self._layers:
            output = layer.forward(output)
        return output


    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        grad_out = grad_z
        for layer in self._layers[::-1]:
            grad_out = layer.backward(grad_out)
        return grad_out


    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag
        self._loss_layer = None
        if self.loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif self.loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()


    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        
        if len(np.shape(input_dataset)) == 1:
            input_dataset = np.reshape(input_dataset,(-1,1))
            #return input_dataset, target_dataset
        
        if len(np.shape(target_dataset)) == 1:
            target_dataset = np.reshape(target_dataset,(-1,1))
        

        indexes = list(range(np.shape(input_dataset)[0]))
        random.shuffle(indexes)
        shuffled_input_dataset = input_dataset[indexes]
        shuffled_target_dataset = target_dataset[indexes]

        # stacked = np.hstack((input_dataset, target_dataset))
        # shuffled_data = np.random.permutation(stacked)
        # shuffled_input_dataset = shuffled_data[:,0:np.shape(input_dataset)[1]]
        # shuffled_target_dataset = shuffled_data[:,np.shape(input_dataset)[1]:]
        return shuffled_input_dataset, shuffled_target_dataset

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        try:
            epoch = 0
            while epoch < self.nb_epoch:
                if self.shuffle_flag == True: 
                    input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
                
                input_batches = np.array_split(input_dataset,self.batch_size)
                target_batches = np.array_split(target_dataset,self.batch_size)

                for i in range(self.batch_size):
                    forward_pass_ = self.network.forward(input_batches[i]) 
                    loss_layer_forward = self._loss_layer.forward(forward_pass_,target_batches[i])
                    grad_z = self._loss_layer.backward()
                    self.network.backward(grad_z)
                    self.network.update_params(self.learning_rate) 

                epoch+=1
        except:
            traceback.print_exc()
            
            
     


    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """

        error = self._loss_layer.forward(self.network.forward(input_dataset), target_dataset)

        return error

class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        self._min = data.min(axis=0)
        self._max = data.max(axis=0)

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        return (data-self._min)/(self._max-self._min)

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        return data*(self._max-self._min)+self._min


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]
    #print("x train")
    print(x_train)
    #print("y_train")
    print(y_train)
    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)
    print("x train pre")
    print(x_train_pre)
    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    #trainer.train(x_train_pre, y_train)
    #print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    #print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))
    trainer.shuffle(x_train_pre, y_train)

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    #print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()

# just testing the Activation Class functions
# x = np.array([[-2, 2, 2], [8, 7, 5], [4, 6, 3]])
# sigmoid = SigmoidLayer()
# A = sigmoid.forward(x)
# B = sigmoid.backward(x)
# print(A)
# print(B)

# relu = ReluLayer()
# Y = relu.forward(x)
# Z = relu.backward(x)
# print(Y)
# print(Z)

# # Testing multi layer network
# network = MultiLayerNetwork(input_dim=3, neurons=[16, 2], activations=["relu", "sigmoid"])
# outputs = network(x)
# print(outputs.shape)
# print(outputs)
# grad_loss_wrt_outputs = np.array([[1, 2], [4, -3], [3, 4]])
# grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs)
# print(grad_loss_wrt_inputs.shape)
# print(grad_loss_wrt_inputs)
# network.update_params(0.01)

# Testing preprocessor
# prep = Preprocessor(x)
# normalised_x = prep.apply(x)
# print(normalised_x)
# original_x = prep.revert(normalised_x)
# print(original_x)


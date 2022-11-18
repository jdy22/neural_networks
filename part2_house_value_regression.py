import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class Regressor():

    def __init__(self, x, nb_epoch = 1000, learning_rate = 0.1, nb_layers = 1, nb_neurons = 23, activation = "tanh"):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - learning_rate {float} -- learning rate for stochastic
                gradient descent.
            - nb_layers {int} -- number of hidden layers in the neural
                network.
            - nb_neurons {int} -- number of neurons per hidden layer.
            - activation {string} -- activation function to apply after
                each hidden layer. "relu", "sigmoid" or "tanh".

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.x =x # for initiallising new regressor in hyperparam tuning 
        self.x_mean = None  # for saving x_mean so that it can be used for testing instances
        self.x_ocean_proximity_mode = None  # for saving ocean_prox_mode so that it can be used for testing instances
        self.y_min = None  # for saving y_min so that it can be used for testing instances
        self.y_max = None  # for saving y_max so that it can be used for testing instances

        # init attributes required for preprocessor only
        self.lb_ocean_proximity = preprocessing.LabelBinarizer()  # init the label binarizer
        self.x_normalizer = preprocessing.MinMaxScaler()  # init the minmax scaler for normalising of x data
        self.y_normalizer = preprocessing.MinMaxScaler()  # init the minmax scaler for normalising of y data

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1] #TODO(Astrid) updated later at pre-processing (see line 126)
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.model = self.Model(self.input_size, self.output_size, self.nb_layers, self.nb_neurons, self.activation)



        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################



        # ensure that regressor has been trained before input testing data set
        if not training and (self.x_mean is None or self.x_ocean_proximity_mode is None):
            return print("Regressor not trained yet")

        # remove the rows that have empty y-values since they cant be used for training
        if training and y is not None: # Added condition for y for when preprocessor is called in __init__ function
            x = x[y.notna().values]
            y = y[y.notna()]

        # handle ocean proximity separately
        x_ocean_proximity = x.loc[:, ["ocean_proximity"]].copy()
        x = x.loc[:, x.columns != "ocean_proximity"].copy()

        # saving mean and mode in training mode + save min and max for y-values during training mode
        if training:
            self.x_mean = x.mean(axis=0) # df has column names, so mean will still be tagged to the indiv feature and not based on index
            self.x_ocean_proximity_mode = x_ocean_proximity.mode()
            if y is not None:
                self.y_min = y.min()
                self.y_max = y.max()

        # filling up NA columns with mean / mode  # TODO(Astrid) justify in our report why choose mean/mode
        x.fillna(self.x_mean, inplace=True)
        x_ocean_proximity.fillna(self.x_ocean_proximity_mode, inplace=True)

        # fit binarizer & normalizer during training
        if training:
            self.lb_ocean_proximity.fit(x_ocean_proximity) # auto updates properties of the binarizer
            self.x_normalizer.fit(x) # auto updates properties of the normalizer for x
            if y is not None:
                self.y_normalizer.fit(y) # auto updates properties of the normalizer for y
                y = pd.DataFrame(self.y_normalizer.transform(y), columns=y.columns) # normalise y values for training instances

        # transform the arrays and convert back into DataFrames
        x_ocean_proximity_onehot = pd.DataFrame(self.lb_ocean_proximity.transform(x_ocean_proximity), columns=self.lb_ocean_proximity.classes_)
        x = pd.DataFrame(self.x_normalizer.transform(x), columns=x.columns)

        # merge back x here with ocean proximity classes columns
        x = pd.concat([x, x_ocean_proximity_onehot], axis=1)

        # Return preprocessed x and y as a tensor, return None for y if it was None
        # Convert x and y to float to use with neural network
        return torch.tensor(x.values).float(), (torch.tensor(y.values).float() if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _postprocessor(self, Y):
        y = self.y_normalizer.inverse_transform(Y)
        return y

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        loss = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for _ in range(self.nb_epoch):
            optimiser.zero_grad()
            predictions = self.model.forward(X)
            mse_loss = loss.forward(input=predictions, target=Y)
            mse_loss.backward()
            optimiser.step()
            print(f"Loss = {mse_loss.item()}")
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        with torch.no_grad():
            X, _ = self._preprocessor(x, training = False) # Do not forget
            output = self.model.forward(X) #TODO (Astrid) scale back the Y predict output
            output = self._postprocessor(output)
            return np.array(output)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        with torch.no_grad():
            X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
            predictions = self.model.forward(X)
            predictions = self._postprocessor(predictions)  #TODO (Astrid) scale back the Y predict output
            return mean_squared_error(np.array(y), np.array(predictions), squared=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    # Functions to make Regressor accomply with GridSearchCV
    def get_params(self, deep=False):
        # return current paramater to the GridSearch
        return {"nb_layers":self.nb_layers, "nb_neurons":self.nb_neurons, "activation":self.activation, "x":self.x,
        "nb_epoch":self.nb_epoch, "learning_rate":self.learning_rate}

    def set_params(self, **parameters):
        # let GridSearch set new params
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    # Inner class defining the neural network
    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size, nb_layers, nb_neurons, activation):
            super().__init__()

            self.input_size = input_size
            self.output_size = output_size
            self.nb_layers = nb_layers
            self.nb_neurons = nb_neurons
            self.activation = activation

            ### Create linear layers of neural network
            # Input layer 
            self.input_layer = torch.nn.Linear(in_features=self.input_size, out_features=self.nb_neurons)
            # Hidden layers
            if self.nb_layers > 1:
                self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(in_features=self.nb_neurons, out_features=self.nb_neurons) for _ in range(self.nb_layers-1)])
            # Output layer
            self.output_layer = torch.nn.Linear(in_features=self.nb_neurons, out_features=self.output_size)

        def forward(self, X):
            """
            Forward pass through neural network.

            Arguments:
                X {torch.tensor} -- Preprocessed input array of shape 
                    (batch_size, input_size).

            Returns:
                {torch.tensor} -- Predicted value for the given input (batch_size, 1).
        
            """
            if self.activation == "relu":
                activation = torch.nn.ReLU()
            elif self.activation == "sigmoid":
                activation = torch.nn.Sigmoid()
            elif self.activation == "tanh":
                activation = torch.nn.Tanh()

            X = self.input_layer(X)
            X = activation(X)

            if self.nb_layers > 1:
                for layer in self.hidden_layers:
                    X = layer(X)
                    X = activation(X)

            # For the output layer apply just the linear transformation
            output = self.output_layer(X)

            return output

def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    grid = {"nb_neurons": np.arange(5, 16), 
        "nb_layers" : np.arange(1, 6),
        "activation": ["relu", "sigmoid", "tanh"],
        "nb_epoch":[500, 1000],
        "learning_rate":[0.1, 0.01, 0.05]
       }

    classifier = GridSearchCV(Regressor(x=x_train), cv=5, param_grid=grid)
    classifier.fit(x_train, y_train)
    print(classifier.best_params_)
    print(classifier.best_score_)
    print(classifier.best_estimator_)
    save_regressor(classifier.best_estimator_)
    return (classifier.best_params_, classifier.best_score_)# Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # split data into testing set and training set
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)

    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # example_main() < excluding this first

    # Testing for pre-processor part:
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    # split data into testing set and training set
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    regressor = Regressor(x_train, nb_epoch = 1000, learning_rate = 0.1, nb_layers = 5, nb_neurons = 7, activation = "tanh")
    x_output, y_output = regressor._preprocessor(x_train, y=y_train, training=True)
    print(x_output)
    print(x_output.shape)
    print(y_output)
    print(y_output.shape)
    x_output_test, _ = regressor._preprocessor(x_train) # trying out for testing mode
    print(x_output_test)
    print(x_output_test.shape)

    print()

    # Testing model training and evaluation:
    print("Testing model training...")
    regressor.fit(x_train, y_train)

    print()

    print("Testing model prediction...")
    prediction = regressor.predict(x_test)
    print(prediction)
    print(prediction.shape)

    print()

    print("Testing model score...")
    score = regressor.score(x_train, y_train)
    print(score)

    # hyperparam tuning, caution very slow
    RegressorHyperParameterSearch(x_train, y_train)


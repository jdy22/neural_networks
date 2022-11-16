import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # X, _ = self._preprocessor(x, training = True) < not sure what's the purpose of this
        self.input_size = x.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.x_mean = None  # for saving x_mean so that it can be used for testing instances
        self.x_ocean_proximity_mode = None # for saving ocean_prox_mode so that it can be used for testing instances
        self.lb_ocean_proximity = preprocessing.LabelBinarizer() # init the label binarizer
        self.x_normalizer = preprocessing.MinMaxScaler() # init the minmax scaler for normalising of data
        return

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

        # # remove the rows that have empty y-values since they cant be used for training
        if training:
            x = x[y.notna().values]
            y = y[y.notna()]

        # handle ocean proximity separately
        x_ocean_proximity = x.loc[:, ["ocean_proximity"]].copy()
        x = x.loc[:, x.columns != "ocean_proximity"].copy()

        # saving mean in training mode
        if training:
            self.x_mean = x.mean(axis=0) # df has column names, so mean will still be tagged to the indiv feature and not based on index
            self.x_ocean_proximity_mode = x_ocean_proximity.mode()

        # filling up NA columns with mean / mode  # TODO(Astrid) justify in our report why choose mean/mode
        x.fillna(self.x_mean, inplace=True)
        x_ocean_proximity.fillna(self.x_ocean_proximity_mode, inplace=True)

        # fit binarizer & normalizer during training
        if training:
            self.lb_ocean_proximity.fit(x_ocean_proximity) # auto updates properties of the binarizer
            self.x_normalizer.fit(x) # auto updates properties of the normalizer

        # transform the arrays and convert back into DataFrames
        x_ocean_proximity_onehot = pd.DataFrame(self.lb_ocean_proximity.transform(x_ocean_proximity), columns=self.lb_ocean_proximity.classes_)
        x = pd.DataFrame(self.x_normalizer.transform(x), columns=x.columns)

        # merge back x here with ocean proximity classes columns
        x = pd.concat([x, x_ocean_proximity_onehot], axis=1)

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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



def RegressorHyperParameterSearch():
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

    return  # Return the chosen hyper parameters

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

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # example_main() < excluding this first

# Testing for pre-processor part:
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = Regressor(x_train, nb_epoch = 10)
    x_output, y_output = regressor._preprocessor(x_train, y=y_train, training=True)
    print(x_output)
    print(y_output)
    x_output_test = regressor._preprocessor(x_train) # trying out for testing mode
    print(x_output_test)






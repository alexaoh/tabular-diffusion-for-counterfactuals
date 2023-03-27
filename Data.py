# Classes for data.

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class Data():
    """Class for pre-processing data. It automatically encodes, splits and scales the data. 
    
    Contains methods for standardization, encoding and train/test/validation splitting.
    
    Parameters
    ----------
    data : dataframe or dict of dataframes
        Pandas df with loaded data or dictionary depending on 'already_splitted_data'.
    cat_features : list of strings.
        List of categorical features. 
    num_features : list of string. 
        List of numerical features. 
    already_splitted_data : Boolean
        (True) If the data is already splitted into train/test/validation, we expect the 'data' parameter 
        to contain a dictionary of {"Train": train_data, "Test": test_data, "Valid": valid_data} 
        (with or without the valid_data depending on value of 'valid').
        (False) If the data is not already splitted, we expect the 'data' parameter to contain a pandas dataframe 
        containing all the data. 
    valid : Boolean 
        True if validation data should be made, False if not. 
    splits : list of real numbers summing to one.
        List of splits for training, testing and validation data, in order. 
    
    Methods 
    -------
    get_training_data_preprocessed :
        Returns a tuple with training data (X,y) which has been preprocessed according to arguments.
    get_training_data :
        Returns a tuple with training data (X,y) before preprocessing.
    get_test_data_preprocessed :
        Returns a tuple with test data (X,y) which has been preprocessed according to arguments.
    get_test_data :
        Returns a tuple with test data (X,y) before preprocessing. 
    get_validation_data_preprocessed :
        Returns a tuple with validation data (X,y) which has been preprocessed according to arguments.
    get_validation_data :
        Returns a tuple with validation data (X,y) before preprocessing.
    train_test_valid_split : 
        Returns a tuple with (X_train, y_train, X_test, y_test) or 
        (X_train, y_train, X_test, y_test, X_valid, y_valid).
    scale : 
        Scale the numerical features according to X_train.
    descale : 
        Descale the numerical features according to X_train.
    fit_scaler :
        Fit sklearn scaler to X_train.
    encode :
        Encode the categorical features according to X_train.
    decode :
        Decode the categorical features according to X_train.
    fit_encoder :
        Fit sklearn encoder to X_train.
    get_original_data :
        Returns the data that was original fed when the object was constructed. 
    find_levels :
        Returns a list of number of levels per categorical feature in the original data fed to the constructor. 
    get_proportion_of_response :
        Returns a pandas Series containing the proportion of each level in the dependent variable.
    """
    def __init__(self, data, cat_features, num_features, already_splitted_data = False, 
                 scale_version = "quantile", valid = True, splits = [0.80, 0.10, 0.10]):
        # The transformations are then done here. 
        self._data = data
        self.categorical_features = cat_features
        self.numerical_features = num_features
        self.already_splitted_data = already_splitted_data
        self.scale_version = scale_version
        self.valid = valid
        assert sum(splits) == 1, "The sum of the splits must be 1!"
        if self.valid:
            assert len(splits) == 3, "You must provide train, test and valid splits."
        else:
            assert len(splits) == 2, "You must provide train and test splits. "
        self.splits = splits
        
        # Assume output always is called 'y'.
        if not already_splitted_data:
            self._X = data.loc[:, self.numerical_features + self.categorical_features]
            self._y = data.loc[:,"y"] 
        else:
            self._X = pd.concat((data["Train"].loc[:, self.numerical_features + self.categorical_features],
                                 data["Test"].loc[:, self.numerical_features + self.categorical_features],
                                 data["Valid"].loc[:, self.numerical_features + self.categorical_features]))
            
            self._y = pd.concat((data["Train"].loc[:, "y"],
                                 data["Test"].loc[:, "y"],
                                 data["Valid"].loc[:, "y"]))

        # Encode the categorical features. 
        if len(self.categorical_features) > 0: # Encode the categorical features if they are provided.
            self.encoder = self.fit_encoder() # Fit the encoder to the categorical data.
            self.X_encoded = self.encode()
        else: # If categorical features are not provided, simply return the data. 
            self.X_encoded = self._X.copy()
        
        # Split into train/test/valid.
        if not already_splitted_data:
            if self.valid:
                (self.X_train, self.y_train, self.X_test, self.y_test, \
                    self.X_valid, self.y_valid) = self.train_test_valid_split(self.X_encoded, self._y)
            else:
                (self.X_train, self.y_train, self.X_test, self.y_test) = self.train_test_valid_split(self.X_encoded, self._y)
        else: 
            train_rows = data["Train"].shape[0]
            test_rows = data["Test"].shape[0]

            self.X_train = self.X_encoded.iloc[:train_rows,:]
            self.y_train = self._y.iloc[:train_rows]

            self.X_test = self.X_encoded.iloc[train_rows:(train_rows+test_rows),:]
            self.y_test = self._y.iloc[train_rows:(train_rows+test_rows)]
            if self.valid: 
                self.X_valid = self.X_encoded.iloc[(train_rows+test_rows):,:]
                self.y_valid = self._y.iloc[(train_rows+test_rows):]

        # Scale the numerical features. 
        if len(self.numerical_features) > 0: # Scale the numerical features if they are provided. 
            self.scaler = self.fit_scaler()
            self.X_train_scaled = self.scale(self.X_train) # Scale the training data.
            self.X_test_scaled = self.scale(self.X_test) # Scale the test data.
            if self.valid:
                self.X_valid_scaled = self.scale(self.X_valid) # Scale the validation data. 
        else: # If numerical features are not provided, simply return the data. 
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
            if self.valid:
                self.X_valid_scaled = self.X_valid.copy()
            
        self.lens_categorical_features = self.find_levels()

    def get_training_data_preprocessed(self):
        """Returns preprocessed training data (X_train, y_train)."""
        return self.X_train_scaled, self.y_train

    def get_training_data(self):
        """Returns training data before preprocessing (X_train_og, y_train)."""
        X_train_og = self.X_train
        if len(self.categorical_features) > 0:
            X_train_og = self.decode(self.X_train)
        return X_train_og, self.y_train
    
    def get_test_data_preprocessed(self):
        """Returns preprocessed test data (X_test, y_test)."""
        return self.X_test_scaled, self.y_test

    def get_test_data(self):
        """Returns test data before preprocessing (X_test_og, y_test)."""
        X_test_og = self.X_test
        if len(self.categorical_features) > 0:
            X_test_og = self.decode(self.X_test)
        return X_test_og, self.y_test
    
    def get_validation_data_preprocessed(self):
        """Returns preprocessed validation data (X_valid, y_valid)."""
        if self.valid:
            return self.X_valid_scaled, self.y_valid
        else: 
            raise ValueError("You did not instantiate this object to contain validation data.")

    def get_validation_data(self):
        """Returns validation data before preprocessing (X_valid_og, y_valid) if applicable."""
        X_valid_og = self.X_valid
        if len(self.categorical_features) > 0:
            X_valid_og = self.decode(self.X_valid)
        if self.valid:
            return X_valid_og, self.y_valid
        else: 
            raise ValueError("You did not instantiate this object to contain validation data.")
    
    def train_test_valid_split(self, X, y):
        """Split data into training/testing/validation, where validation is optional at instantiation."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.splits[0], random_state=42)
        if self.valid:
            sum_test_valid = self.splits[1]+self.splits[2]
            sum_train_valid = self.splits[0]+self.splits[2]
            X_test, X_valid, y_test, y_valid = train_test_split( \
                                        X_test, y_test, train_size=round((1.0-sum_train_valid)/sum_test_valid, 5), random_state=42)
            return (X_train, y_train, X_test, y_test, X_valid, y_valid) # Round above because of floating point issues. 
        return (X_train, y_train, X_test, y_test)
            
    def scale(self, df):
        """Scale the numerical features according to the TRAINING data."""
        output = df.copy() # Deep copy the given df. 
        output[self.numerical_features] = self.scaler.transform(output[self.numerical_features])
        return output
        
    def descale(self, df):
        """Descale the numerical features according to the TRAINING data."""
        output = df.copy()
        output[self.numerical_features] = self.scaler.inverse_transform(output[self.numerical_features])
        return output

    def fit_scaler(self):
        """Fit the scaler to the numerical TRAINING data. Supports StandardScaler and QuantileTransformer."""
        if self.scale_version == "standard":
            return preprocessing.StandardScaler().fit(self.X_train[self.numerical_features])
        elif self.scale_version == "quantile":
            return preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(self.X_train.shape[0] // 30, 1000), 10),
            subsample=1e9).fit(self.X_train[self.numerical_features])
        else:
            raise NotImplementedError(f"The scaler '{self.scale_version}' has not been implemented.")
    
    def encode(self):
        """Encode the categorical data. Only supports OneHotEncoding."""
        output = self._X.copy() # Deep copy the X-data.
        encoded_features = self.encoder.get_feature_names(self.categorical_features) # Get the encoded names. 
        
        # Add the new columns to the new dataset (all the levels of the categorical features).
        output[encoded_features] = self.encoder.transform(output[self.categorical_features])

        # Remove the old columns (before one-hot encoding)
        output = output.drop(self.categorical_features, axis = 1) 
        return output
    
    def decode(self, df):
        """Decode the categorical data. Only support OneHotEncoding."""
        output = df.copy()
        column_names = output.columns.tolist()
        encoded_features = self.encoder.get_feature_names(self.categorical_features) # Get the encoded names. 
        
        if len(encoded_features) == 0:
            return output # Does not work when there are not categorical features in df.
        
        output[self.categorical_features] = self.encoder.inverse_transform(output[encoded_features])
        output = output.drop(encoded_features, axis=1)
        if "y" in column_names:
            return output[self._X.columns.tolist() + ["y"]] # Reorder the columns to match the original order of the dataframe. 
        else:
            return output[self._X.columns] # Reorder the columns to match the original order of the dataframe. 

    def fit_encoder(self):
        """Fit the encoder to the categorical data. Only supports OneHotEncoding."""
        return preprocessing.OneHotEncoder(handle_unknown = "ignore", \
          sparse = False, drop = None).fit(self._X[self.categorical_features])

    def get_original_data(self):
        """Returns the original data as fed to the class."""
        if not self.already_splitted_data:
            return self._data
        else:
            data = self._X.copy()
            data["y"] = self._y
            return data

    def find_levels(self):
        """Returns a list of levels of features of each of the categorical features."""
        df = self._X.copy()
        lens_categorical_features = []
        for feat in self.categorical_features:
            unq = len(df[feat].value_counts().keys().unique())
            #print(f"Feature '{feat}'' has {unq} unique levels")
            lens_categorical_features.append(unq)
        #print(f"The sum of all levels is {sum(lens_categorical_features)}. This will be the number of cat-columns after one-hot encoding (non-full rank)")
        return(lens_categorical_features)
    
    def get_proportion_of_response(self, normalize = True):
        """Returns the proportions of the response in the entire data set."""
        return self._y.value_counts(normalize=normalize)
    
class CustomDataset(Dataset):
    """Class for using data with Pytorch."""
    def __init__(self, X, y, transform = None):
        self.X = X.values.astype(np.float32) # Return a Numpy array with the dataframe contents. 
        self.y = np.reshape(y.values,(len(y.values),1)).astype(np.float32) # Return a Numpy array with the dataframe contents. 

        self.n_samples = self.X.shape[0]
        self.transform = transform
       
    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
   
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    """Callable object to transform CustomDataset inputs and labels to Pytorch tensors."""
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)

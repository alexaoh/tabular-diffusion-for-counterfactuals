# Another attempt at implementing a VAE in Pytorch
# following the blog https://avandekleut.github.io/vae/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

import pandas as pd
import numpy as np

# Configure the device 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using '{device}' device.")

# Set some seeds.
#seed = 0
#np.random.seed(seed)
#torch.manual_seed(seed)

class VariationalEncoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.l1 = nn.Linear(input_size, 15)
        self.l2 = nn.Linear(15, latent_dim) 
        self.l3 = nn.Linear(15, latent_dim)
        # Above we add two identical output layers for the encoder, 
        # where one is for mu and the other is for log_var.
        
        self.relu = nn.ReLU() # Activation function.

        # Other utilities. 
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda() # Hack to get sampling on the GPU.
        self.N.scale = self.N.scale.cuda() # Hack to get sampling on the GPU.
        self.KL = 0
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        mu = self.l2(x)
        log_var = self.l3(x)
        z = mu + torch.exp(0.5*log_var)*self.N.sample(mu.shape)
        self.KL = -0.5*(1 + log_var - torch.exp(log_var) - mu**2).sum()
        return z # Could also return mu, std and log_var, then calculate z later. 

class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim, len_numerical_features, lens_categorical_features):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(latent_dim, 15)
        #self.l2 = nn.Linear(15, input_size)
        self.l2 = nn.Linear(15, len_numerical_features)
        self.cat_output_layers = nn.ModuleList()
        for i, val in enumerate(lens_categorical_features):
            self.cat_output_layers.append(nn.Linear(15, val))
        # I don't think it is enough with only two!
        # I need one output layer per categorical variable, with the number of levels as nodes per layer.

        self.relu = nn.ReLU() # Activation function. 

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        #x = self.l2(x)
        x_num = self.l2(x)
        x_cat = [] # List of output layers for each categorical feature.
        for lay in self.cat_output_layers:
            x_cat.append(lay(x))
        return x_num, x_cat

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, len_numerical_features, lens_categorical_features):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim, len_numerical_features, lens_categorical_features)
        assert isinstance(lens_categorical_features, list), "'lens_categorical_features' is meant to be a list of each of the levels in the categorical features."
        self.lens_categorical_features = lens_categorical_features
        self.len_numerical_features = len_numerical_features

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, train_data_loader, epochs = 30, device = "cuda"):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.01)
    train_losses = []  
    n_total_steps = len(train_data_loader) # Total length of training data. 
    cat_loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (inputs, _) in enumerate(train_data_loader):
            # Split the inputs into categorical and numerical.
            # The first batch of features are numerical and the second batch of features are categorical. 
            inputs = inputs.to(device) # Load data onto device.
            inputs_num = inputs[:,:autoencoder.len_numerical_features]

            # Burde gjøre denne oppsplittingen i Dataloaderen (data-objektet mitt) i stedet!
            # Kan nok ganske enkelt og greit bare flytte denne dit senere. 
            lens_cumprod = np.cumsum(autoencoder.lens_categorical_features)
            lens_cumprod = np.insert(lens_cumprod, 0, 0, axis = 0) # insert 0 first for index trick in loop below. 
            inputs_cat = [] # Perhaps I don't really need the inputs?
            labels_cat = [] 
            for j, val in enumerate(lens_cumprod[1:]):
                first_index = autoencoder.len_numerical_features + lens_cumprod[j]
                final_index = autoencoder.len_numerical_features + val
                values = inputs[:,first_index:final_index]
                inputs_cat.append(values)
                labels_cat.append(torch.argmax(values, dim = 1)) # Reverse one-hot encode the labels. 
            # Inputs cat is a list over inputs per categorical feature. 
            # Labels cat is a list over the backward one-hot encoded inputs (showing the correct level per categorical feature).

            # We make tensors from the arrays and put them on the device.
            labels_cat = torch.stack(labels_cat) # shape = [categorical_features, batches]
            # Not sure if this actually was necessary actually but it seems to work fine below.

            #inputs_cat = inputs[:,autoencoder.len_numerical_features:]
            # Reverse one-hot encode inputs_cat before using CrossEntropy as loss!
            #labels_cat = torch.argmax(inputs_cat, dim = 1)
            
            x_hat_num, x_hat_cats = autoencoder(inputs)

            #loss = ((inputs - x_hat)**2).sum() + autoencoder.encoder.KL

            loss_numerical = ((inputs_num - x_hat_num)**2).sum()
            
            losses_categorical = []
            for j, val in enumerate(x_hat_cats):
                # Sjekk de her igjen, om de gir mening!!
                l = cat_loss(val, labels_cat[j])
                losses_categorical.append(l)
            
            #loss_categorical = cat_loss(x_hat_cat, labels_cat)
            loss_KL = autoencoder.encoder.KL
            
            loss = loss_numerical + sum(losses_categorical) + loss_KL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Could also simply print the loss per epoch, instead of for some prespecified iteration number!
            if (i+1) % 500 == 0:
                train_losses.append(loss.item())
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    return autoencoder, train_losses


# Load some data and try to train the thing. 
# from Data import Data, CustomDataset, ToTensor

# # Load the data.
# adult_data = pd.read_csv("adult_data_no_NA.csv", index_col = 0)
# #print(adult_data.shape) # Looks good!

# categorical_features = ["workclass","marital_status","occupation","relationship", \
#                         "race","sex","native_country"]
# numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

# # We reorder adult_data such that all the numerical features come first, then the categorical features. 
# # The label 'y' comes finally. This will be our assumption about the data set when fitting the model.
# adult_data = adult_data[numerical_features + categorical_features + ["y"]]


# def find_levels(df, categorical_features):
#     """Returns a list of levels of features of each of the categorical features."""
#     lens_categorical_features = []
#     for feat in categorical_features:
#         unq = len(df[feat].value_counts().keys().unique())
#         print(f"Feature '{feat}'' has {unq} unique levels")
#         lens_categorical_features.append(unq)
#     print(f"The sum of all levels is {sum(lens_categorical_features)}. This will be the number of cat-columns after one-hot encoding (non-full rank)")
#     return(lens_categorical_features)

# lens_categorical_features = find_levels(adult_data.loc[:,adult_data.columns != "y"], categorical_features)
# print(lens_categorical_features)


# # Make Adult Dataclass. 
# Adult = Data(adult_data, categorical_features, numerical_features, valid = True)

# # Load the preprocessed data. 
# X_train_prep, y_train = Adult.get_training_data_preprocessed()
# X_test_prep, y_test = Adult.get_test_data_preprocessed()
# X_valid_prep, y_valid = Adult.get_validation_data_preprocessed()

# # Load the original data to have for later. 
# #X_train_og, _  = Adult.get_training_data()
# #X_test_og, _  = Adult.get_test_data()
# #X_valid_og, _ = Adult.get_validation_data()

# # Checking my new additions to the Data-class. 
# #X_train_og.to_csv("train_data_checking.csv")
# #X_test_og.to_csv("test_data_checking.csv")
# #X_valid_og.to_csv("valid_data_checking.csv")

# # Make training data for Pytorch. 
# train_data = CustomDataset(X_train_prep, y_train, transform = ToTensor())

# # Set some hyperparameters.
# batch_size = 16 
# input_size = X_train_prep.shape[1]
# latent_dim = 2

# # Define the training data loader. 
# train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)

# # Define the autoencoder.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# autoencoder = VAE(input_size, latent_dim, len(numerical_features), lens_categorical_features).to(device)

# # Train the autoencoder.
# autoencoder, train_losses = train(autoencoder, train_data_loader, epochs = 10)

# Plot the losses. 
def plot_loss(train_losses):
    plt.plot(train_losses)
    plt.title("Training Loss per 'Iter'")
    plt.xlabel("'Iter'")
    plt.ylabel("Loss")
    plt.show()

#plot_loss(train_losses)

def plot_latent(autoencoder, data_loader):
    """Plot the two-dimensional latent space."""
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader): # Go through all batches of training data. 
            z = autoencoder.encoder(x.to(device))
            z = z.cpu().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.show()

# We plot the latent representation in 2D of the training data. 
#plot_latent(autoencoder, train_data_loader)

# Try to sample some synthetic data from the model. 
# Could/should synthesize from a gaussian for the VAE instead of doing it "deterministically" like below. 
def synthesize(autoencoder):
    """Generate some synthetic data from a deterministic autoencoder. Assuming a 2D latent space."""
    x_n = y_n = 100
    x_points = np.linspace(-4, 3, x_n)
    y_points = np.linspace(-3, 5, y_n)

    saver = np.empty((x_n*y_n, X_train_prep.shape[1]))
    saver[:] = np.nan
    i = 0
    with torch.no_grad():
        for x in x_points:
            for y in y_points:
                z = torch.Tensor([[x,y]]).to(device)
                x_hat = autoencoder.decoder(z)
                saver[i,:] = x_hat.cpu().numpy()
                i+=1
    df = pd.DataFrame(saver, columns = X_train_prep.columns.tolist())
    return df

#synthetic_data = synthesize(autoencoder)
#synthetic_data_de = Adult.decode(synthetic_data)
#synthetic_data_de = Adult.descale(synthetic_data_de)
#synthetic_data_de.to_csv("synth_vae.csv")

#import seaborn as sbs
#sbs.pairplot(synthetic_data_de)
#plt.show()

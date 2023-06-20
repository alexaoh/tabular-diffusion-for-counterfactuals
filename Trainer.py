# Classes for training Gaussian_diffusion, Multinomial_diffusion and Gaussian_multinomial_diffusion models. 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Data import Data, CustomDataset, ToTensor

class Trainer():
    """Trainer for diffusion models. This is a base class that the more specific trainers inherit from.
    
    Parameters
    ----------
    data : dict
        Dictionary containing training and validation data.
        Expected shape:
            data = {
                "X_train": X_train,
                "y_train": y_train,
                "X_valid": X_valid, 
                "y_valid": y_valid
            }
    model : Neural network to be trained. 
        NeuralNet Object we want to train.
    epochs : int
        Number of epochs to train for.
    batch_size : int
        Batch size to use during training.
    learning_rate : float
        Learning rate for the optimizer (default is Adam).  
    early_stop_tolerance : int
        Number of epochs in tolerance before early stopping is called. 
        Set equal to 'None' if you want to turn early stopping off. 

    Methods 
    -------
    train :
        Main training loop to call when you want to train the model.
    plot_losses :
        Plot losses after training. 
    """
    def __init__(self, data, model, epochs, batch_size, learning_rate, early_stop_tolerance = 15, data_code = "AD"):
        self.data = data
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_tolerance = early_stop_tolerance
        self.data_code = data_code # Used for saving the trained models. 

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_valid = data["X_valid"]
        self.y_valid = data["y_valid"]

        self.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        # Arrays for saving losses each epoch.
        self.training_losses = np.zeros(epochs)
        self.validation_losses = np.zeros(epochs)

        # Data loaders for training and validation.
        train_data = CustomDataset(self.X_train, self.y_train, transform = ToTensor())         
        valid_data = CustomDataset(self.X_valid, self.y_valid, transform = ToTensor()) 
        self.train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
        self.valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = 2) # We validate on the same batch_size as during training. 

    def train(self):
        """Main training loop."""
        pass

    def plot_losses(self):
        """Plot losses after training."""
        print(f"Length of non-zero training losses: {len(self.training_losses[self.training_losses != 0])}")
        print(f"Length of non-zero validation losses: {len(self.validation_losses[self.validation_losses != 0])}")
        plt.plot(self.training_losses[self.training_losses != 0], label = "Training")
        plt.plot(self.validation_losses[self.validation_losses != 0], label = "Validation")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

class Gaussian_trainer(Trainer):
    """Trainer for Gaussian_diffusion."""

    def __init__(self, data, model, gaussian_diffusion, epochs, batch_size, learning_rate, early_stop_tolerance=15, data_code = "AD"):
        super().__init__(data, model, epochs, batch_size, learning_rate, early_stop_tolerance, data_code)
        self.gaussian_diffusion = gaussian_diffusion
        self.numerical_features = gaussian_diffusion.numerical_features
        self.device = gaussian_diffusion.device

    def train(self):
        """Main training loop."""

        min_valid_loss = np.inf # For early stopping and saving model. 
        count_without_improving = 0 # For early stopping.

        for epoch in range(self.epochs):
            self.model.train()
            self.gaussian_diffusion.train() # Strictly not necessary, but good practice. 
            
            train_loss = 0.0
            for i, (inputs,y) in enumerate(self.train_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
                t = self.gaussian_diffusion.sample_timesteps(inputs.shape[0]).to(self.device)

                # Noise the inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
                x_t, noise = self.gaussian_diffusion.noise_data_point(inputs, t) # x_t is the noisy version of the input x, at time t.

                # Feed the noised data and the time step to the model, which then predicts the noise at that time. 
                predicted_noise = self.model(x_t, t, y)

                # Gaussian diffusion uses MSE loss. 
                loss = self.gaussian_diffusion.loss(noise, predicted_noise)

                self.optimizer.zero_grad()
                loss.backward() # Calculate gradients. 
                self.optimizer.step() # Update parameters. 
                train_loss += loss.item() # Calculate total training loss over the entire epoch.

            train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.  
            
            ######################### Validation.
            self.model.eval()
            self.gaussian_diffusion.eval()
            valid_loss = 0.0
            for i, (inputs, y) in enumerate(self.valid_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # We sample new times. 
                t = self.gaussian_diffusion.sample_timesteps(inputs.shape[0]).to(self.device)

                # We noise the validation inputs at the new sampled times. 
                x_t, noise = self.gaussian_diffusion.noise_data_point(inputs, t) 

                predicted_noise = self.model(x_t, t, y) # Predict the noise of the validation data. 

                # Gaussian diffusion uses MSE loss. 
                loss = self.gaussian_diffusion.loss(noise, predicted_noise)
                valid_loss += loss # Calculate the sum of validation loss over the entire epoch.

            valid_loss = valid_loss / (i+1)
            #########################         

            self.training_losses[epoch] = train_loss
            self.validation_losses[epoch] = valid_loss
            
            print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}. Validation loss after epoch {epoch+1} is {valid_loss:.4f}.")
            
            # Saving models each time the validation loss reaches a new minimum.
            if min_valid_loss > valid_loss:
                print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
                
                min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

                # Saving the new "best" models. 
                extra = ""
                if not self.model.is_class_cond:
                    extra = "_joint"            
                torch.save(self.gaussian_diffusion.state_dict(), "pytorch_models/"+self.data_code+"_Gaussian_diffusion"+extra+str(self.model.seed)+".pth")
                torch.save(self.model.state_dict(), "pytorch_models/"+self.data_code+"_Gaussian_diffusion_Neural_net"+extra+str(self.model.seed)+".pth")
                count_without_improving = 0
            else:
                count_without_improving += 1

            if self.early_stop_tolerance is not None:
                # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
                if count_without_improving >= int(self.early_stop_tolerance):
                    print("Early stopping triggered")
                    return 

class Multinomial_trainer(Trainer):
    """Trainer for Multinomial_diffusion."""

    def __init__(self, data, model, multinomial_diffusion, epochs, batch_size, learning_rate, early_stop_tolerance=15, data_code = "AD"):
        super().__init__(data, model, epochs, batch_size, learning_rate, early_stop_tolerance, data_code)
        self.multinomial_diffusion = multinomial_diffusion
        self.categorical_feature_names = multinomial_diffusion.categorical_feature_names
        self.categorical_levels = multinomial_diffusion.categorical_levels
        self.device = multinomial_diffusion.device

    def train(self):
        """Main training loop."""
    
        min_valid_loss = np.inf # For early stopping and saving model. 
        count_without_improving = 0 # For early stopping.

        for epoch in range(self.epochs):
    
            # Set PyTorch objects to training mode. Not necessary for all configurations, but good practice. 
            self.model.train()
            self.multinomial_diffusion.train()

            train_loss = 0.0
            for i, (inputs, y) in enumerate(self.train_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # Assuming the data is already one-hot-encoded, we do a log-transformation of the data. 
                log_inputs = torch.log(inputs.float().clamp(min=1e-30))

                # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
                t, pt = self.multinomial_diffusion.sample_timesteps(inputs.shape[0])
                t = t.to(self.device)
                pt = pt.to(self.device)
                # Running it without the prior now, according to loss in the multinomial diffusion class.
                
                # Noise the inputs and return the noised input in log_space.
                log_x_t = self.multinomial_diffusion.forward_sample(log_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
                log_predictions = self.model(log_x_t, t, y)

                # Multinomial diffusion uses KL for discrete quantities. 
                loss = self.multinomial_diffusion.loss(log_inputs, log_x_t, log_predictions, t, pt)

                self.optimizer.zero_grad()
                loss.backward() # Calculate gradients. 
                self.optimizer.step() # Update parameters. 
                train_loss += loss.item() # Calculate total training loss over the entire epoch.

            train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.  
            
            ######################### Validation.
            # Set PyTorch objects to evaluation mode. Not necessary for all configurations, but good practice.
            self.model.eval()
            self.multinomial_diffusion.eval()

            valid_loss = 0.0
            for i, (inputs,y) in enumerate(self.valid_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # Assuming the data is already one-hot-encoded, we do a log-transformation of the data. 
                log_inputs = torch.log(inputs.float().clamp(min=1e-30))

                # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
                t, pt = self.multinomial_diffusion.sample_timesteps(inputs.shape[0])
                t = t.to(self.device)
                pt = pt.to(self.device)
                # Running it without the prior now, according to loss in the multinomial diffusion class.
                
                # Noise the inputs and return the noised input in log_space.
                log_x_t = self.multinomial_diffusion.forward_sample(log_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
                log_predictions = self.model(log_x_t, t, y)

                # Multinomial diffusion uses KL for discrete quantities. 
                loss = self.multinomial_diffusion.loss(log_inputs, log_x_t, log_predictions, t, pt)
                valid_loss += loss.item()

            valid_loss = valid_loss / (i+1)
            #########################         

            self.training_losses[epoch] = train_loss
            self.validation_losses[epoch] = valid_loss
            
            print(f"Training loss after epoch {epoch+1} is {train_loss:.4f}. Validation loss after epoch {epoch+1} is {valid_loss:.4f}.")
            
            # Saving models each time the validation loss reaches a new minimum.
            if min_valid_loss > valid_loss:
                print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
                
                min_valid_loss = valid_loss # Set new minimum validation loss. 

                # Saving the new "best" models.   
                extra = ""
                if not self.model.is_class_cond:
                    extra = "_joint"          
                torch.save(self.multinomial_diffusion.state_dict(), "pytorch_models/"+self.data_code+"_Multinomial_diffusion"+extra+str(self.model.seed)+".pth")
                torch.save(self.model.state_dict(), "pytorch_models/"+self.data_code+"_Multinomial_diffusion_Neural_net"+extra+str(self.model.seed)+".pth")
                count_without_improving = 0
            else:
                count_without_improving += 1

            if self.early_stop_tolerance is not None:
                # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
                if count_without_improving >= int(self.early_stop_tolerance):
                    print("Early stopping triggered.")
                    return 

class Gaussian_multinomial_trainer(Trainer):
    """Trainer for Gaussian_multinomial_diffusion."""

    def __init__(self, data, model, multinomial_diffusion, gaussian_diffusion, 
                 epochs, batch_size, learning_rate, early_stop_tolerance=15, data_code = "AD"):
        super().__init__(data, model, epochs, batch_size, learning_rate, early_stop_tolerance, data_code)
        self.gaussian_diffusion = gaussian_diffusion
        self.multinomial_diffusion = multinomial_diffusion
        self.device = gaussian_diffusion.device

        # Add extra arrays for the two different types of losses. 
        self.gaussian_losses = np.zeros(epochs)
        self.multinomial_losses = np.zeros(epochs)

    def train(self):
        """Main training loop."""
        num_numerical_features = len(self.gaussian_diffusion.numerical_features)
        num_categorical_features = len(self.multinomial_diffusion.categorical_feature_names)

        min_valid_loss = np.inf # For early stopping and saving model. 
        count_without_improving = 0 # For early stopping.

        for epoch in range(self.epochs):
    
            # Set PyTorch objects to training mode. Not necessary for all configurations, but good practice. 
            self.model.train()
            self.gaussian_diffusion.train() 
            self.multinomial_diffusion.train()

            train_loss = 0.0
            gaussian_loss = 0.0
            multinomial_loss = 0.0
            for i, (inputs, y) in enumerate(self.train_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # Split the input data into numerical and categorical covariates. 
                gauss_inputs = inputs[:,:num_numerical_features]
                mult_inputs = inputs[:, num_numerical_features:]
                log_mult_inputs = torch.log(mult_inputs.float().clamp(min=1e-40)) # We effectively change all zeros to a very small number using "clamp", to avoid log(0).

                # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
                t, pt = self.multinomial_diffusion.sample_timesteps(inputs.shape[0])
                t = t.to(self.device)
                pt = pt.to(self.device)
                
                # Noise the numerical inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
                x_t_gauss, noise_gauss = self.gaussian_diffusion.noise_data_point(gauss_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Noise the categorical inputs.
                log_x_t_mult = self.multinomial_diffusion.forward_sample(log_mult_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Concatenate the inputs for the neural network. 
                x_t = torch.cat((x_t_gauss, log_x_t_mult), dim = 1)

                # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
                predictions = self.model(x_t, t, y)

                predicted_noise = predictions[:, :num_numerical_features]
                predicted_log_probs = predictions[:, num_numerical_features:]

                # Gaussian diffusion uses MSE loss. 
                gauss_loss = self.gaussian_diffusion.loss(noise_gauss, predicted_noise)

                # Multinomial diffusion uses KL for discrete quantities. 
                mult_loss = self.multinomial_diffusion.loss(log_mult_inputs, log_x_t_mult, predicted_log_probs, t, pt) / num_categorical_features 

                # Calculate total loss. Downweigh the multinomial diffusion loss by the number of categorical features. 
                loss = gauss_loss + mult_loss 

                self.optimizer.zero_grad()
                loss.backward() # Calculate gradients. 
                self.optimizer.step() # Update parameters. 
                train_loss += loss.item() # Calculate total training loss over the entire epoch.
                gaussian_loss += gauss_loss.item()
                multinomial_loss += mult_loss.item()

            train_loss = train_loss / (i+1) # Divide the training loss by the number of batches. 
                                            # In this way we make sure the training loss and validation loss are on the same scale.  

            gaussian_loss = gaussian_loss / (i+1)
            multinomial_loss = multinomial_loss / (i+1)
            
            ######################### Validation.
            # Set PyTorch objects to evaluation mode. Not necessary for all configurations, but good practice.
            self.model.eval()
            self.gaussian_diffusion.eval() 
            self.multinomial_diffusion.eval()

            valid_loss = 0.0
            for i, (inputs,y) in enumerate(self.valid_loader):
                # Load data onto device.
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.to(torch.int) # Make sure the labels are integers (0,1,...)

                # Split the input data into numerical and categorical covariates. 
                gauss_inputs = inputs[:,:num_numerical_features]
                mult_inputs = inputs[:, num_numerical_features:]
                log_mult_inputs = torch.log(mult_inputs.float().clamp(min=1e-40)) # We effectively change all zeros to a very small number using "clamp", to avoid log(0).

                # Sample random timesteps between 1 and 'noise_steps' (uniformly) for diffusion process.
                t, pt = self.multinomial_diffusion.sample_timesteps(inputs.shape[0])
                t = t.to(self.device)
                pt = pt.to(self.device)
                # Running it without the prior now, according to loss in the multinomial diffusion class.
                
                # Noise the numerical inputs and return the noise. This noise is important when calculating the loss, since we want to predict this noise as closely as possible. 
                x_t_gauss, noise_gauss = self.gaussian_diffusion.noise_data_point(gauss_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Noise the categorical inputs.
                log_x_t_mult = self.multinomial_diffusion.forward_sample(log_mult_inputs, t) # x_t is the noisy version of the input x, at time t.

                # Concatenate the inputs for the neural network. 
                x_t = torch.cat((x_t_gauss, log_x_t_mult), dim = 1)

                # Feed the noised data and the time step to the model, which then predicts the noise (for numerical) and log_prob (for categorical).
                predictions = self.model(x_t, t, y)

                predicted_noise = predictions[:, :num_numerical_features]
                predicted_log_probs = predictions[:, num_numerical_features:]

                # Gaussian diffusion uses MSE loss. 
                gauss_loss = self.gaussian_diffusion.loss(noise_gauss, predicted_noise)

                # Multinomial diffusion uses KL for discrete quantities. 
                mult_loss = self.multinomial_diffusion.loss(log_mult_inputs, log_x_t_mult, predicted_log_probs, t, pt) / num_categorical_features 

                # Calculate total loss. Downweigh the multinomial diffusion loss by the number of categorical features. 
                loss = gauss_loss + mult_loss 
                valid_loss += loss # Calculate the sum of validation loss over the entire epoch.

            valid_loss = valid_loss / (i+1)
            #########################         

            self.training_losses[epoch] = train_loss
            self.validation_losses[epoch] = valid_loss
            self.gaussian_losses[epoch] = gaussian_loss
            self.multinomial_losses[epoch] = multinomial_loss
            
            print(f"Epoch {epoch+1}: GLoss: {gaussian_loss:.4f}, MLoss: {multinomial_loss:.4f}, Total: {train_loss:.4f}. VLoss: {valid_loss:.4f}.")
            
            # Saving models each time the validation loss reaches a new minimum.
            if min_valid_loss > valid_loss:
                print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
                
                min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

                # Saving the new "best" models.             
                extra = ""
                if not self.model.is_class_cond:
                    extra = "_joint"
                torch.save(self.gaussian_diffusion.state_dict(), 
                           "pytorch_models/"+self.data_code+"_Gaussian_multinomial_diffusion_Gaussian_part"+extra+str(self.model.seed)+".pth")
                torch.save(self.multinomial_diffusion.state_dict(), 
                           "pytorch_models/"+self.data_code+"_Gaussian_multinomial_diffusion_Multinomial_part"+extra+str(self.model.seed)+".pth")
                torch.save(self.model.state_dict(), "pytorch_models/"+self.data_code+"_Gaussian_multinomial_diffusion_Neural_net"+extra+str(self.model.seed)+".pth")
                count_without_improving = 0
            else:
                count_without_improving += 1

            if self.early_stop_tolerance is not None:
                # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
                if count_without_improving >= int(self.early_stop_tolerance):
                    print("Early stopping triggered")
                    return 
    
    def plot_losses(self):
        """Plot losses after training."""
        print(f"Length of non-zero training losses: {len(self.training_losses[self.training_losses != 0])}")
        print(f"Length of non-zero validation losses: {len(self.validation_losses[self.validation_losses != 0])}")
        plt.plot(self.training_losses[self.training_losses != 0], label = "Training")
        plt.plot(self.validation_losses[self.validation_losses != 0], label = "Validation")
        plt.plot(self.gaussian_losses[self.gaussian_losses != 0], label = "Gaussian")
        plt.plot(self.multinomial_losses[self.multinomial_losses != 0], label = "Multinomial")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

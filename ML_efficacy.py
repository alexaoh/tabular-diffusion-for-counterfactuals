# We check ML efficacy of synthetic data with a simple NN.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics # plot_roc_curve.

from Data import Data, CustomDataset, ToTensor

class SimpleNeuralNetClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleNeuralNetClassifier, self).__init__()
        self.input_size = input_size
        
        # Layers. 
        self.l1 = nn.Linear(input_size, 18)
        self.l2 = nn.Linear(18,9)
        self.l3 = nn.Linear(9,3)
        self.output = nn.Linear(3,1)
        
        # Activation functions.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.output(out)
        out = self.sigmoid(out) 
        return out
    

def train(model, X_train, y_train, X_valid, y_valid, batch_size, num_epochs, device, savename = "SimpleNeuralNetClassAdultSynth"):
    """Training loop for simple classifier."""
    input_size = X_train.shape[1] # Columns in the training data is the input size of the neural network model. 

    # Make PyTorch dataset. 
    train_data = CustomDataset(X_train, y_train, transform = ToTensor())         
    valid_data = CustomDataset(X_valid, y_valid, transform = ToTensor()) 

    # Make train_data_loader for batching, etc in Pytorch.
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(valid_data, batch_size = X_valid.shape[0], num_workers = 2) # We want to validate on the entire validation set in each epoch

    # Define the optimizer. 
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Define the loss for the binary classifier.
    criterion = nn.BCELoss() # Loss function.

    # Main training loop.
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    training_accuracies = np.zeros(num_epochs)
    validation_accuracies = np.zeros(num_epochs)

    min_valid_loss = np.inf # For early stopping and saving model. 
    count_without_improving = 0 # For early stopping.

    for epoch in range(num_epochs):
        # Set PyTorch objects to training mode. Not necessary for all configurations, but good practice. 
        model.train()

        train_loss = 0.0
        train_accuracy = 0.0

        for i, (inputs, labels) in enumerate(train_loader):  
            # Load the data on to the gpu.
            inputs = inputs.to(device)
            y_acc_metric = labels.reshape(-1).numpy() # Used to measure accuracy as a metric below. 
            labels = labels.view(labels.shape[0],1).to(device) 
            
            # Forward pass.
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accuracy metric reporting. 
            accuracy = (outputs.reshape(-1).cpu().detach().numpy().round() == y_acc_metric).mean()    

            # Backward and optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_accuracy += accuracy
        
        # Calculate means over the epoch. 
        train_loss = train_loss / i
        train_accuracy = train_accuracy / i

        model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0

        for i, (inputs, labels) in enumerate(valid_loader):
             # Load the data on to the gpu.
            inputs = inputs.to(device)
            y_acc_metric = labels.reshape(-1).numpy() # Used to measure accuracy as a metric below. 
            labels = labels.view(labels.shape[0],1).to(device) 
            
            # Forward pass.
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accuracy metric reporting. 
            accuracy = (outputs.reshape(-1).cpu().detach().numpy().round() == y_acc_metric).mean()   

            valid_loss += loss
            valid_accuracy += accuracy

        training_losses[epoch] = train_loss
        training_accuracies[epoch] = train_accuracy
        validation_losses[epoch] = valid_loss
        validation_accuracies[epoch] = valid_accuracy

        print(f"Epoch {epoch+1}: TLoss: {train_loss:.4f}, VLoss: {valid_loss:.4f}, TAcc: {train_accuracy:.4f}. VAcc: {valid_accuracy:.4f}.")
        
        # Saving models each time the validation loss reaches a new minimum.
        if min_valid_loss > valid_loss:
            print(f"Validation loss decreased from {min_valid_loss:.4f} to {valid_loss:.4f}. Saving the model.")
            
            min_valid_loss = valid_loss.item() # Set new minimum validation loss. 

            # Saving the new "best" model.
            torch.save(model.state_dict(), "./"+savename+".pth")
            count_without_improving = 0
        else:
            count_without_improving += 1

        # Early stopping. Return the losses if the model does not improve for a given number of consecutive epochs. 
        if count_without_improving >= 10:
            return training_losses, validation_losses, training_accuracies, validation_accuracies
        
    return training_losses, validation_losses, training_accuracies, validation_accuracies

def test(model, X_test, y_test, device):
    """Test on entire test set at once."""
    with torch.no_grad():
        test_dat = torch.from_numpy(X_test.values.astype(np.float32)).to(device)
        test_labs = torch.from_numpy(y_test.values.astype(np.float32))
        test_labs = test_labs.view(test_labs.shape[0],1)
        
        predicted_probs = model(test_dat).cpu().numpy()
        
        #plt.hist(predicted_probs.flatten())
        #plt.xlim(0,1)
        print(f"Number of predictions: {len(predicted_probs.flatten())}")
        print(f"Unique predicted values: {np.unique(predicted_probs.flatten())}, length: {len(np.unique(predicted_probs.flatten()))}")
        return predicted_probs

def plot_losses(training_losses, validation_losses, label1 = "Training", label2 = "Validation"):
    print(f"Length of non-zero training losses: {len(training_losses[training_losses != 0])}")
    print(f"Length of non-zero validation losses: {len(validation_losses[validation_losses != 0])}")
    plt.plot(training_losses[training_losses != 0], label = label1)
    plt.plot(validation_losses[validation_losses != 0], label = label2)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.legend()
    #plt.show()

def main():
    """Main function to run if file is called directly."""
    # Load data etc here. 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' device.")

    # Set seeds for reproducibility. 
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    categorical_features = ["workclass","marital_status","occupation","relationship", \
                            "race","sex","native_country"]
    numerical_features = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]

    # Load the real data into the scope. 
    adult_data = pd.read_csv("adult_data_no_NA.csv", index_col = 0)
    print(adult_data.shape)

    # Load the synthetic data into the scope. 
    synthetic_samples = pd.read_csv("synthetic_sample_both.csv", index_col = 0)
    print(synthetic_samples.shape)

    Adult = Data(adult_data, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train, y_train = Adult.get_training_data_preprocessed()
    X_valid, y_valid = Adult.get_validation_data_preprocessed()
    X_test, y_test = Adult.get_test_data_preprocessed()
    print(X_train.shape)

    lens_categorical_features = Adult.lens_categorical_features

    # Synthetic data har ikke labels, hvordan kan jeg trene den da?
    # Må generere falske labels sammen med den syntetiske dataen også!
    # Synth = Data(synthetic_samples, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    # X_train_s, y_train_s = Synth.get_training_data_preprocessed()
    # X_valid_s, y_valid_s = Synth.get_validation_data_preprocessed()
    # X_test_s, y_test_s = Synth.get_test_data_preprocessed()
    # print(X_train_s.shape)

    classifier = SimpleNeuralNetClassifier(X_train.shape[1]).to(device)
    summary(classifier)

    # Hyperparameters. 
    batch_size = 128
    epochs = 100

    # Train the simple classifier on the true data.
    adult_train_losses, adult_valid_losses, adult_train_accuracies, adult_valid_accuracies = \
        train(model=classifier, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, 
              batch_size=batch_size, num_epochs=epochs, device=device, savename = "SimpleNeuralNetClassAdult")
    
    plot_losses(adult_train_losses, adult_valid_losses)
    print(adult_train_accuracies)
    print(adult_valid_accuracies)
    plt.show()

    # Test the simple classifier trained on true data on true test set.
    predicted_probs_adult = test(model=classifier, X_test=X_test, y_test=y_test, device=device)

    # Make confusion matrix.
    labs = list(y_test.values)
    preds = predicted_probs_adult.flatten()
    predicted_classes = np.where(preds > 0.5, 1, 0)
    cm = metrics.confusion_matrix(labs, list(predicted_classes), labels = [0,1])
    conf_mat = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    conf_mat.plot()
    plt.show()

    print("Some more classifaction statistics:")
    print(metrics.classification_report(labs, predicted_classes, labels = [0,1]))

    # Make roc and auc.
    fpr, tpr, thresholds = metrics.roc_curve(labs, predicted_probs_adult.flatten())
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='Simple Neural Net Classifier')
    display.plot()
    plt.show()

if __name__ == "__main__":
    main()

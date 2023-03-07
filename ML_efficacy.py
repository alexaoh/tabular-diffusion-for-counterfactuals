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

def train(model, X_train, y_train, X_valid, y_valid, batch_size, num_epochs, device, savename = "AD_SimpleNeuralNetClassAdultSynth"):
    """Training loop for simple classifier."""

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
            torch.save(model.state_dict(), "./pytorch_models/ML_efficacy/"+savename+".pth")
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
        test_labs = torch.from_numpy(y_test.values.astype(np.float32)).to(device)
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

def make_confusion_matrix(y_test, predicted_probs_true_data, predicted_probs_synth):
    """Make and plot confusion matrix for both the models."""
    labs = list(y_test.values)
    preds_true = predicted_probs_true_data.flatten()
    preds_synth = predicted_probs_synth.flatten()
    predicted_classes_true = np.where(preds_true > 0.5, 1, 0)
    predicted_classes_synth = np.where(preds_synth > 0.5, 1, 0)

    cm_true = metrics.confusion_matrix(labs, list(predicted_classes_true), labels = [0,1])
    conf_mat_true = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_true)

    cm_synth = metrics.confusion_matrix(labs, list(predicted_classes_synth), labels = [0,1])
    conf_mat_synth = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_synth)

    fig, ax = plt.subplots(1,2)
    conf_mat_true.plot(ax = ax[0], colorbar = False)
    ax[0].set_title("AD Real Data")
    conf_mat_synth.plot(ax = ax[1])
    ax[1].set_title("AD Synthetic Data")
    plt.show()

    print("Some more classifaction statistics:")
    print(metrics.classification_report(labs, predicted_classes_true, labels = [0,1]))
    print(metrics.classification_report(labs, predicted_classes_synth, labels = [0,1]))

def calculate_auc_f1(y_test, predicted_probs_true_data, predicted_probs_synth):
    """Calculate metrics we want to use to compare ML efficacy with."""
    labs = list(y_test.values)
    preds_true = predicted_probs_true_data.flatten()
    preds_synth = predicted_probs_synth.flatten()
    predicted_classes_true = np.where(preds_true > 0.5, 1, 0)
    predicted_classes_synth = np.where(preds_synth > 0.5, 1, 0)

    fpr_true, tpr_true, _ = metrics.roc_curve(labs, preds_true)
    auc_true = metrics.auc(fpr_true, tpr_true)

    display_true = metrics.RocCurveDisplay(fpr=fpr_true, tpr=tpr_true, roc_auc=auc_true,
                                    estimator_name='AD Real Data')
    
    f1_true = metrics.f1_score(labs, predicted_classes_true)

    fpr_synth, tpr_synth, _ = metrics.roc_curve(labs, preds_synth)
    auc_synth = metrics.auc(fpr_synth, tpr_synth)

    display_synth = metrics.RocCurveDisplay(fpr=fpr_synth, tpr=tpr_synth, roc_auc=auc_synth,
                                    estimator_name='AD Synthetic Data')
    
    f1_synth = metrics.f1_score(labs, predicted_classes_synth)

    fig, ax = plt.subplots(1,2)
    display_true.plot(ax = ax[0])
    display_synth.plot(ax = ax[1])
    ax[0].set_title("AD Real Data")
    ax[1].set_title("AD Synthetic Data")
    plt.show()

    return f1_true, auc_true, f1_synth, auc_synth

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
    training = pd.read_csv("splitted_data/AD/AD_train.csv", index_col = 0)
    testing = pd.read_csv("splitted_data/AD/AD_test.csv", index_col = 0)
    valid = pd.read_csv("splitted_data/AD/AD_valid.csv", index_col = 0)
    data = {"Train":training, "Test":testing, "Valid":valid}

    Data_object = Data(data, cat_features = categorical_features, num_features = numerical_features,
                            already_splitted_data=True, scale_version="quantile", valid = True)
    X_train, y_train = Data_object.get_training_data_preprocessed()
    X_test, y_test = Data_object.get_test_data_preprocessed()
    X_valid, y_valid = Data_object.get_validation_data_preprocessed()
    print(f"X_train.shape: {X_train.shape}")

    adult_data = Data_object.get_original_data()
    print(f"adult_data.shape: {adult_data.shape}")

    # Load the synthetic data into the scope. 
    synthetic_samples = pd.read_csv("synthetic_data/AD_Gaussian_multinomial_diffusion.csv", index_col = 0)
    print(f"synthetic_samples.shape: {synthetic_samples.shape}")

    Synth = Data(synthetic_samples, categorical_features, numerical_features, scale_version = "quantile", valid = True)
    X_train_s, y_train_s = Synth.get_training_data_preprocessed()
    X_valid_s, y_valid_s = Synth.get_validation_data_preprocessed()
    print(f"X_train_synthetic.shape: {X_train_s.shape}")

    classifier_real = SimpleNeuralNetClassifier(X_train.shape[1]).to(device)
    summary(classifier_real)

    classifier_synth = SimpleNeuralNetClassifier(X_train_s.shape[1]).to(device)

    # Hyperparameters. 
    batch_size = 128
    epochs = 100

    # Train the simple classifier on the true data.
    adult_train_losses, adult_valid_losses, adult_train_accuracies, adult_valid_accuracies = \
        train(model=classifier_real, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, 
              batch_size=batch_size, num_epochs=epochs, device=device, savename = "AD_SimpleNeuralNetClassReal")
    
    plot_losses(adult_train_losses, adult_valid_losses)
    #print(adult_train_accuracies)
    #print(adult_valid_accuracies)
    plt.show()

    # Train the simple classifier on the synthetic data.
    synth_train_losses, synth_valid_losses, synth_train_accuracies, synth_valid_accuracies = \
        train(model=classifier_synth, X_train=X_train_s, y_train=y_train_s, X_valid=X_valid_s, y_valid=y_valid_s, 
              batch_size=batch_size, num_epochs=epochs, device=device, savename = "AD_SimpleNeuralNetClassSynth")
    
    plot_losses(synth_train_losses, synth_valid_losses)
    #print(synth_train_accuracies)
    #print(synth_valid_accuracies)
    plt.show()

    # Test the simple classifier trained on true data on true test set.
    predicted_probs_real = test(model=classifier_real, X_test=X_test, y_test=y_test, device=device)
    
    # Test the simple classifier trained on synthetic data on true test set.
    predicted_probs_synth = test(model=classifier_synth, X_test=X_test, y_test=y_test, device=device)

    # Plot classification matrix and print some more stats.
    make_confusion_matrix(y_test, predicted_probs_real, predicted_probs_synth)

    # Calculate f1 score and auc, and return these values.
    f1_true, auc_true, f1_synth, auc_synth = calculate_auc_f1(y_test, predicted_probs_real, predicted_probs_synth)

    print(f"F1 score from real data: {f1_true}")
    print(f"F1 score from synthetic data: {f1_synth}")
    print(f"AUC from real data: {auc_true}")
    print(f"AUC from synthetic data {auc_synth}")

if __name__ == "__main__":
    main()

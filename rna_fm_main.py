import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel, BertConfig
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from multimolecule import RnaTokenizer, RnaFmModel

from train import train_model
from utils.plot import plot_scatter, plot_confusion_matrix, plot_loss
from utils.load_data import load_data
from utils.cls_eff import classify_efficiency
from models.model_rnafm import FullModel

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#--------------Please edit this part to change the model and hyperparameters---------#
# Define output name
outputname = "rnafm_experiment"  # Change this to your desired output name

# Load data
batch_size = 32
# ------------------------------
# Load RNAFM and Tokenize
# ------------------------------

# Initialize the tokenizer
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")

# Disable bos and eos tokens if not used
tokenizer.bos_token = None
tokenizer.eos_token = None



train_loader, tokens1_test, tokens2_test, eff_test = load_data("pentamer5MLx.csv", tokenizer, LLM_type='RNA', undersample_high_eff=0.3, test_size=0.15, batch_size=batch_size, device=device)

# Define model and training parameters
input_size = 640 * 2
n_epochs = 250
n_splits = 5
#------------end of editable part--------------#


kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation loop
fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
    print(f"Fold {fold + 1}/{n_splits}")

    # Create train and validation subsets
    train_subset = Subset(train_loader.dataset, train_idx)
    val_subset = Subset(train_loader.dataset, val_idx)
    train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    #--------------Please edit this part to change the model and hyperparameters---------#
    # Reconstruc the model every fold
    # Load RNA-FM model every fold
    rna_fm = RnaFmModel.from_pretrained("multimolecule/rnafm")

    # Unfreeze RNAFM parameters if they were frozen
    for param in rna_fm.parameters():
        param.requires_grad = True

    model = FullModel(rna_fm, input_size)
    model = model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    #------------end of editable part--------------#


    # Train the model
    best_model_state, training_loss, validation_loss = train_model(
        train_loader_fold, val_loader_fold, model, criterion, optimizer, scheduler, n_epochs, device=device
    )

    # Load the best model state
    model.load_state_dict(best_model_state)
    model.eval()

    # Evaluate on the test dataset
    with torch.no_grad():
        y_pred = model(tokens1_test, tokens2_test)
        test_loss = criterion(y_pred, eff_test.unsqueeze(1)).item()
        y_pred_cpu = y_pred.cpu().numpy()
        eff_test_cpu = eff_test.cpu().numpy()

    # Convert predicted and true efficiency values into classification groups
    y_pred_class = np.array([classify_efficiency(val) for val in y_pred_cpu.flatten()])
    y_test_class = np.array([classify_efficiency(val) for val in eff_test_cpu.flatten()])

    # Calculate classification accuracy
    classification_accuracy = accuracy_score(y_test_class, y_pred_class)
    print(f"Classification accuracy for fold {fold + 1} is: {classification_accuracy:.4f}")

    # Save results
    fold_results.append({
        "fold": fold + 1,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "best_val_loss": min(validation_loss),
        "test_loss": test_loss,
        "classification_accuracy": classification_accuracy,
    })

    # Plot scatter plot
    plot_scatter(y_pred_cpu, eff_test_cpu, fold + 1, outputname)

    # Plot confusion matrix
    plot_confusion_matrix(y_test_class, y_pred_class, fold + 1, outputname)

    # Plot training and validation loss
    plot_loss(training_loss, validation_loss, fold + 1, outputname)

    # Save the best model for this fold
    torch.save(best_model_state, f'{outputname}_best_model_fold_{fold + 1}.pth')

    # Clear memory
    torch.cuda.empty_cache()

# Save fold results
with open(f'{outputname}_fold_results.json', 'w') as f:
    json.dump(fold_results, f, indent=4)

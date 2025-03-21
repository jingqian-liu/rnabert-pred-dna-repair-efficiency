import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_scatter(y_pred, y_true, fold, outputname):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_true)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '--k')
    plt.xlabel("Predicted Efficiency")
    plt.ylabel("True Efficiency")
    plt.title(f"Fold {fold}: Predicted vs True Efficiency")
    plt.savefig(f'{outputname}_fold_{fold}_scatter.pdf')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, fold, outputname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Fold {fold}: Confusion Matrix')
    plt.savefig(f'{outputname}_fold_{fold}_confusion_matrix.pdf')
    plt.show()

def plot_loss(training_loss, validation_loss, fold, outputname):
    plt.figure()
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold}: Training and Validation Loss")
    plt.savefig(f'{outputname}_fold_{fold}_loss.pdf')
    plt.show()

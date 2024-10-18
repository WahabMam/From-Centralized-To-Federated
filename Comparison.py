import matplotlib.pyplot as plt
import numpy as np

# Centralized and Federated learning accuracies
centralized_acc = [65.69, 68.72, 68.36]  # Accuracies on entire test set for Model 1, Model 2, Model 3
centralized_unseen_acc = [0.0, 0.0, 0.0]  # Accuracies on unseen digits [1,3,7], [2,5,8], [4,6,9]

federated_acc = [95.94, 97.35, 94.48, 94.64]  # Accuracies from the federated learning process
federated_labels = ['All Digits', '[1,3,7]', '[2,5,8]', '[4,6,9]']

# Plot the accuracy comparison
def plot_accuracy_comparison():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Centralized Learning Plot
    ax[0].bar(['Model 1', 'Model 2', 'Model 3'], centralized_acc, color='blue', label='Entire Dataset')
    ax[0].bar(['Model 1', 'Model 2', 'Model 3'], centralized_unseen_acc, color='red', label='Unseen Digits')
    ax[0].set_ylim(0, 100)
    ax[0].set_title('Centralized Learning')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].legend(loc='upper left')

    # Federated Learning Plot
    ax[1].bar(federated_labels, federated_acc, color='green', label='Federated Results')
    ax[1].set_ylim(0, 100)
    ax[1].set_title('Federated Learning')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].legend(loc='upper left')

    plt.suptitle('Centralized vs Federated Learning Accuracy Comparison')
    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap='Blues')
    plt.colorbar(cax)
    ax.set_title(f'{title} Confusion Matrix', pad=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Assuming you have confusion matrices for centralized and federated learning
# Centralized and Federated confusion matrices can be calculated similarly to your evaluation step
cm_centralized = np.array([[50, 2, 1], [1, 45, 5], [0, 3, 52]])  # Example for Centralized
cm_federated = np.array([[58, 1, 0], [1, 57, 2], [0, 1, 59]])  # Example for Federated

def compare_confusion_matrices():
    plot_confusion_matrix(cm_centralized, "Centralized Model")
    plot_confusion_matrix(cm_federated, "Federated Model")

if __name__ == "__main__":
    plot_accuracy_comparison()
    compare_confusion_matrices()

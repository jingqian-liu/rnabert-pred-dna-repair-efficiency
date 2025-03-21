import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
# Each list contains accuracy values for different folds for a specific LLM
dnabert_accuracies = [ 0.41626794258373206, 0.4019138755980861, 0.39712918660287083, 0.2966507177033493, 0.5645933014354066]  # Example accuracies for DNABERT
rnabert_accuracies = [ 0.8947368421052632, 0.9043062200956937, 0.8708133971291866, 0.84688995215311, 0.8564593301435407]  # Example accuracies for RNABERT
rnafm_accuracies = [0.8851674641148325, 0.9186602870813397, 0.861244019138756, 0.9138755980861244, 0.8564593301435407]    # Example accuracies for RNAFM

# Calculate mean and standard deviation for each LLM
dnabert_mean = np.mean(dnabert_accuracies)
dnabert_std = np.std(dnabert_accuracies)

rnabert_mean = np.mean(rnabert_accuracies)
rnabert_std = np.std(rnabert_accuracies)

rnafm_mean = np.mean(rnafm_accuracies)
rnafm_std = np.std(rnafm_accuracies)

# Calculate mean and standard deviation for each LLM
dnabert_mean = np.mean(dnabert_accuracies)
dnabert_std = np.std(dnabert_accuracies)

rnabert_mean = np.mean(rnabert_accuracies)
rnabert_std = np.std(rnabert_accuracies)

rnafm_mean = np.mean(rnafm_accuracies)
rnafm_std = np.std(rnafm_accuracies)

# Create x-axis labels and corresponding mean/std values
llm_types = ["DNABERT", "RNABERT", "RNAFM"]
means = [dnabert_mean, rnabert_mean, rnafm_mean]
stds = [dnabert_std, rnabert_std, rnafm_std]

# Create the scatter plot
plt.figure(figsize=(2, 6))

# Plot individual fold accuracies for each LLM
for i, (llm, accuracies) in enumerate(zip(llm_types, [dnabert_accuracies, rnabert_accuracies, rnafm_accuracies])):
    # Scatter plot for individual fold accuracies
    plt.scatter([i] * len(accuracies), accuracies, color = ['cornflowerblue', 'violet', 'paleturquoise'][i], label=f'{llm} Folds', alpha=0.8)

    # Plot mean accuracy with error bars
    plt.errorbar(i, means[i], yerr=stds[i], fmt='o', color=['cornflowerblue', 'violet', 'paleturquoise'][i], capsize=5, label=f'{llm} Mean ± Std' if i == 0 else "")

# Add labels and title
plt.xlabel("LLM Type", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
#plt.title("Accuracy of Different LLMs Across 5 Folds", fontsize=14)
plt.xticks(range(len(llm_types)), llm_types)  # Set x-axis ticks and labels
plt.ylim(0.2, 1.0)  # Adjust y-axis limits if needed

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot

# Save the plot
plt.savefig("llm_accuracy_scatter_plot_with_folds.svg", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

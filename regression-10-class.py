import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import RnaTokenizer, RnaBertModel
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
df = pd.read_csv("pentamer5MLx.csv", dtype={"column_name": str})[["Strand1(5>3)", "Strand2(3>5)", "Mean"]]
sequences_1 = df["Strand1(5>3)"].to_list()
sequences_2 = df["Strand2(3>5)"].to_list()
eff = torch.tensor(df["Mean"].astype(float).values, dtype=torch.float)

print("Sample sequences:", sequences_1[-4:-2])

# Change T to U in sequences
rna_sequences_1 = [seq.replace('T', 'U') for seq in sequences_1]
rna_sequences_2 = [seq.replace('T', 'U') for seq in sequences_2]
print("RNA Sample sequences:", rna_sequences_1[-4:-2])

# Reverse the second strand (3'>5' to 5'>3')
rna_reversed_sequences_2 = [seq[::-1] for seq in rna_sequences_2]

# ------------------------------
# Load RNABert and Tokenize
# ------------------------------
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
rna_bert = RnaBertModel.from_pretrained("multimolecule/rnabert")

# Set bos and eos tokens to None if not used
tokenizer.bos_token = None
tokenizer.eos_token = None

# Split raw sequences before tokenization
seq1_train, seq1_test, seq2_train, seq2_test, eff_train, eff_test = train_test_split(
    rna_sequences_1, rna_reversed_sequences_2, eff, test_size=0.3
)

# Tokenize train and test sequences separately
tokens1_train = tokenizer(seq1_train, return_tensors="pt", padding=True, truncation=True)
tokens2_train = tokenizer(seq2_train, return_tensors="pt", padding=True, truncation=True)
tokens1_test  = tokenizer(seq1_test, return_tensors="pt", padding=True, truncation=True)
tokens2_test  = tokenizer(seq2_test, return_tensors="pt", padding=True, truncation=True)

# Move tokenized data to GPU
tokens1_train = {k: v.to('cuda') for k, v in tokens1_train.items()}
tokens2_train = {k: v.to('cuda') for k, v in tokens2_train.items()}
tokens1_test  = {k: v.to('cuda') for k, v in tokens1_test.items()}
tokens2_test  = {k: v.to('cuda') for k, v in tokens2_test.items()}

# Unfreeze RNABert parameters if they were frozen
for param in rna_bert.parameters():
    param.requires_grad = True

# ------------------------------
# Define Model Components
# ------------------------------
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.W_query = nn.Parameter(torch.rand(120, 120))  # Weight matrix for queries
        self.W_key   = nn.Parameter(torch.rand(120, 120))  # Weight matrix for keys
        self.W_value = nn.Parameter(torch.rand(120, 120))  # Weight matrix for values

    def forward(self, x_1, x_2, attn_mask=None):
        query = torch.matmul(x_1, self.W_query)
        key   = torch.matmul(x_2, self.W_key)
        value = torch.matmul(x_2, self.W_value)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_attn_scores = attn_scores / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scaled_attn_scores = scaled_attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class FullModel(nn.Module):
    def __init__(self, rna_bert, input_size):
        super(FullModel, self).__init__()
        self.rna_bert = rna_bert  # Pretrained RNABert
        self.cross_attention1 = CrossAttention()
        self.cross_attention2 = CrossAttention()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens1, tokens2):
        # Get embeddings from RNABert
        embedded_seq1 = self.rna_bert(**tokens1).last_hidden_state
        embedded_seq2 = self.rna_bert(**tokens2).last_hidden_state

        # Optionally apply cross-attention (currently not used in final prediction)
        x1, _ = self.cross_attention1(embedded_seq1, embedded_seq2)
        x2, _ = self.cross_attention2(embedded_seq2, embedded_seq1)

        # Average token embeddings along sequence length dimension
        s1_avg = torch.mean(embedded_seq1, dim=1)
        s2_avg = torch.mean(embedded_seq2, dim=1)

        # Concatenate averaged embeddings and pass through MLP for prediction
        x = torch.cat((s1_avg, s2_avg), dim=1)
        return self.mlp(x)

# Instantiate the full model
input_size = 120 * 2  # 120 is the embedding dimension from RNABert (example)
model = FullModel(rna_bert, input_size)
print(model)
model = model.cuda()

# ------------------------------
# Define Dataset for Tokenized Inputs
# ------------------------------
class MismatchDataset(Dataset):
    def __init__(self, tokens1, tokens2, efficiency):
        """
        Args:
            tokens1: Tokenized data (dict) for first sequence.
            tokens2: Tokenized data (dict) for second sequence.
            efficiency: Corresponding efficiency values.
        """
        self.tokens1 = tokens1
        self.tokens2 = tokens2
        self.efficiency = efficiency

    def __len__(self):
        return self.efficiency.size(0)

    def __getitem__(self, idx):
        # Extract the idx-th element from each tensor in the dictionary
        item1 = {k: v[idx] for k, v in self.tokens1.items()}
        item2 = {k: v[idx] for k, v in self.tokens2.items()}
        y = self.efficiency[idx]
        return item1, item2, y

bs = 16
train_set = MismatchDataset(tokens1_train, tokens2_train, eff_train)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 200
training_loss = []
test_loss_all = []
test_loss_best = np.inf
criterion = nn.L1Loss()

# ------------------------------
# Training and Evaluation Loop
# ------------------------------
for i in range(n_epochs):
    print("Epoch " + str(i+1))
    model.train()
    epoch_loss = 0.0

    for bn, (inputs1, inputs2, targets) in enumerate(train_loader):
        # Move targets to GPU
        targets = targets.to(dtype=torch.float32).cuda()
        # Forward pass
        outputs = model(inputs1, inputs2)
        optimizer.zero_grad()
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()  # Remove retain_graph=True unless needed
        optimizer.step()
        epoch_loss += loss.item()

    avg_training_loss = epoch_loss / len(train_loader)
    training_loss.append(avg_training_loss)
    print('Epoch %d training loss: %.4f' % (i+1, avg_training_loss))

    # Evaluate on test data (using the already GPU-resident tokens)
    model.eval()
    with torch.no_grad():
        test_outputs = model(tokens1_test, tokens2_test)
        eff_test = eff_test.to(dtype=torch.float32).cuda()
        test_loss = criterion(test_outputs, eff_test.unsqueeze(1))
        test_loss_all.append(test_loss.item())

    if test_loss < test_loss_best:
        test_loss_best = test_loss
        #torch.save(model.state_dict(), os.path.join('./', 'best_regression.pth'))

    print('Epoch %d testing loss: %.4f' % (i+1, test_loss.item()))

# Save the final model
#torch.save(model.state_dict(), os.path.join('./', 'final_regression.pth'))

# Plot training and testing loss
plt.plot(training_loss, label="Training Loss")
plt.plot(test_loss_all, label="Test Loss")
plt.legend()
plt.savefig('loss.pdf')


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


model.eval()
with torch.no_grad():
    y_pred = model(tokens1_test, tokens2_test)

# Function to map efficiency value to classification group
def classify_efficiency(eff):
    # Assuming the mapping: 0-0.5 -> group 0; 0.5-0.8 -> group 1; >0.8 -> group 3
    if eff <= 0.1:
        return 0
    elif eff <= 0.2:
        return 1
    elif eff <= 0.3:
        return 2
    elif eff <= 0.4:
        return 3
    elif eff <= 0.5:
        return 4
    elif eff <= 0.6:
        return 5
    elif eff <= 0.7:
        return 6
    elif eff <= 0.8:
        return 7
    elif eff <= 0.9:
        return 8
    else:
        return 9

plt.figure(figsize=(6, 6))
ax = plt.gca()  # Make sure to include the parentheses!

y_pred_cpu = y_pred.cpu().numpy()
eff_test_cpu = eff_test.cpu().numpy()
plt.scatter(y_pred_cpu, eff_test_cpu)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1],[0,1],'--k')

# Set equal aspect ratio so that x and y scales are equal
ax.set_aspect('equal', adjustable='box')

plt.xlabel('Predicted efficiency')
plt.ylabel('Ground truth efficiency')
plt.savefig('eff_pred.pdf')
plt.show()



# Convert the predicted and true efficiency values into classification groups
y_pred_class = np.array([classify_efficiency(val) for val in y_pred])
y_test_class = np.array([classify_efficiency(val) for val in eff_test])

# Calculate the accuracy and confusion matrix of the classification
accuracy = accuracy_score(y_test_class, y_pred_class)
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

print("Classification Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:")
print(conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0','1','2','3','4','5','6','7','8','9'],
            yticklabels=['0','1','2','3','4','5','6','7','8','9'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


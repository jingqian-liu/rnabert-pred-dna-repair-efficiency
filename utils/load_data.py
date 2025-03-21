import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

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



def load_data(csv_file, tokenizer, LLM_type='RNA', undersample_high_eff = 0.3, test_size = 0.15, batch_size=32, device='cuda'):
    # Load the dataset and select relevant columns
    df = pd.read_csv(csv_file, dtype={"column_name": str})[["Strand1(5>3)", "Strand2(3>5)", "Mean"]]
    df["Mean"] = df["Mean"].astype(float)

    # Split into high- and low-efficiency groups, then downsample the high-efficiency set
    df_high_eff = df[df["Mean"] > 0.8]
    df_low_eff = df[df["Mean"] <= 0.8]
    df_high_eff_sampled = df_high_eff.sample(frac=undersample_high_eff, random_state=42)
    df_balanced = pd.concat([df_low_eff, df_high_eff_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract sequence lists and efficiency tensor
    sequences_1 = df_balanced["Strand1(5>3)"].to_list()
    sequences_2 = df_balanced["Strand2(3>5)"].to_list()
    eff = torch.tensor(df_balanced["Mean"].values, dtype=torch.float)


    if LLM_type == 'RNA':
        # Convert DNA to RNA (T -> U) and reverse the second strand (3'>5' to 5'>3')
        rna_sequences_1 = [seq.replace('T', 'U') for seq in sequences_1]
        rna_sequences_2 = [seq.replace('T', 'U') for seq in sequences_2]
        rna_reversed_sequences_2 = [seq[::-1] for seq in rna_sequences_2]

        # Split raw sequences into training and test sets
        seq1_train, seq1_test, seq2_train, seq2_test, eff_train, eff_test = train_test_split(
            rna_sequences_1, rna_reversed_sequences_2, eff, test_size=test_size
        )

    elif LLM_type == 'DNA':
        # Reverse the second strand (3'>5' to 5'>3')
        reversed_sequences_2 = [seq[::-1] for seq in sequences_2]

        # Split raw sequences into training and test sets
        seq1_train, seq1_test, seq2_train, seq2_test, eff_train, eff_test = train_test_split(
            sequences_1, reversed_sequences_2, eff, test_size=test_size
        )

    else:
        print("Wrong input for LLM type!")

    # Tokenize sequences separately for training and testing
    tokens1_train = tokenizer(seq1_train, return_tensors="pt", padding=True, truncation=True)
    tokens2_train = tokenizer(seq2_train, return_tensors="pt", padding=True, truncation=True)
    tokens1_test  = tokenizer(seq1_test, return_tensors="pt", padding=True, truncation=True)
    tokens2_test  = tokenizer(seq2_test, return_tensors="pt", padding=True, truncation=True)

    # Move tokenized data to the specified device (GPU)
    tokens1_train = {k: v.to(device) for k, v in tokens1_train.items()}
    tokens2_train = {k: v.to(device) for k, v in tokens2_train.items()}
    tokens1_test  = {k: v.to(device) for k, v in tokens1_test.items()}
    tokens2_test  = {k: v.to(device) for k, v in tokens2_test.items()}

    # Return a DataLoader for training and test data
    train_set = MismatchDataset(tokens1_train, tokens2_train, eff_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Return test tokens separately along with test efficiency tensor
    eff_test = eff_test.to(dtype=torch.float32).to(device)
    
    return train_loader, tokens1_test, tokens2_test, eff_test


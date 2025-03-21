import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FullModel(nn.Module):
    def __init__(self, rna_bert, input_size):
        super(FullModel, self).__init__()
        self.rna_bert = rna_bert  # Pretrained RNABert
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tokens1, tokens2):
        # Get embeddings from RNABert
        embedded_seq1 = self.rna_bert(**tokens1).last_hidden_state
        embedded_seq2 = self.rna_bert(**tokens2).last_hidden_state


        # Average token embeddings along sequence length
        s1_avg = torch.mean(embedded_seq1, dim=1)
        s2_avg = torch.mean(embedded_seq2, dim=1)

        # Concatenate and pass through MLP
        x = torch.cat((s1_avg, s2_avg), dim=1)
        return self.mlp(x)


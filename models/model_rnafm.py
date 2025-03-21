import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        # Using fixed dimensions (e.g., 120) for demonstration
        self.W_query = nn.Parameter(torch.rand(640, 640))
        self.W_key   = nn.Parameter(torch.rand(640, 640))
        self.W_value = nn.Parameter(torch.rand(640, 640))

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
    def __init__(self, rna_fm, input_size):
        """
        Args:
            rna_fm: A pretrained RNA model (e.g., RnaFmModel)
            input_size: Dimension of concatenated averaged embeddings (e.g., 640*2)
        """
        super(FullModel, self).__init__()
        self.rna_fm = rna_fm  # Pretrained RNA model

        # Uncomment below to use cross-attention modules if needed
        #self.cross_attention = CrossAttention()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # Optionally, add activation (e.g., nn.Sigmoid()) if output normalization is needed
        )

    def forward(self, tokens1, tokens2):
        # Get embeddings from the pretrained RNA model
        embedded_seq1 = self.rna_fm(**tokens1).last_hidden_state
        embedded_seq2 = self.rna_fm(**tokens2).last_hidden_state


        # Average token embeddings along sequence dimension
        s1_avg = torch.mean(embedded_seq1, dim=1)
        s2_avg = torch.mean(embedded_seq2, dim=1)

        # Concatenate averaged embeddings and feed into MLP for regression output
        x = torch.cat((s1_avg, s2_avg), dim = 1)

        return self.mlp(x)


    def get_embeddings(self, tokens1, tokens2):
        """
        Compute and return the embeddings for the input tokens.
        This method is used during evaluation to save embeddings.
        """
        # Get embeddings from the pretrained RNA model
        embedded_seq1 = self.rna_fm(**tokens1).last_hidden_state
        embedded_seq2 = self.rna_fm(**tokens2).last_hidden_state


        # Average token embeddings along sequence dimension
        s1_avg = torch.mean(embedded_seq1, dim=1)
        s2_avg = torch.mean(embedded_seq2, dim=1)

        # Concatenate averaged embeddings
        embeddings = torch.cat((s1_avg, s1_avg), dim=1)

        return embeddings


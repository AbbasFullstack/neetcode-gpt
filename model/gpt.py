import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim, head_size):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(model_dim, head_size, bias=False)
        self.query_gen = nn.Linear(model_dim, head_size, bias=False)
        self.value_gen = nn.Linear(model_dim, head_size, bias=False)

    def forward(self, embedded):
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)
        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)
        lower_triangular = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)
        return scores @ v


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        torch.manual_seed(0)
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(SingleHeadAttention(model_dim, model_dim // num_heads))
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, embedded):
        head_outputs = []
        for head in self.att_heads:
            head_outputs.append(head(embedded))
        concatenated = torch.cat(head_outputs, dim=2)
        return self.output_proj(concatenated)


class VanillaNeuralNetwork(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        torch.manual_seed(0)
        self.up_projection = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.ReLU()
        self.down_projection = nn.Linear(model_dim * 4, model_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # FIX: seed yahan se hataya gaya - dropout ka mask deterministic ban raha tha
        return self.dropout(self.down_projection(self.relu(self.up_projection(x))))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        torch.manual_seed(0)
        self.attention = MultiHeadedSelfAttention(model_dim, num_heads)
        self.linear_network = VanillaNeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded):
        # FIX: seed yahan se hataya gaya
        embedded = embedded + self.attention(self.first_norm(embedded))
        embedded = embedded + self.linear_network(self.second_norm(embedded))
        return embedded


class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, model_dim, num_blocks, num_heads):
        super().__init__()
        torch.manual_seed(0)  # seed sirf __init__ mein - weights reproducible, dropout live rehta hai
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(model_dim, num_heads))
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        # FIX: seed yahan se hataya gaya
        # FIX: torch.round(logits) hataya - round ka gradient 0 hota hai,
        # isliye loss.backward() model ko kuch sikha nahi raha tha
        embedded = self.word_embeddings(context)
        positions = torch.arange(context.shape[1], device=context.device)
        embedded = embedded + self.position_embeddings(positions)
        output = self.final_norm(self.transformer_blocks(embedded))
        logits = self.vocab_projection(output)
        return logits

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, nhead=4, num_layers=2,
                 hidden_dim=128, output_dim=3, dropout=0.2, pad_idx=None,
                 context_vocab_size=None, pretrained_embeddings=None):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.context_embedding = None
        if context_vocab_size is not None:
            self.context_embedding = nn.Embedding(context_vocab_size, embed_dim, padding_idx=pad_idx)
            self.context_fc = nn.Linear(embed_dim, hidden_dim)
            self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim + (hidden_dim if self.context_embedding is not None else 0), output_dim)

    def forward(self, text, context=None):
        embedded = self.embedding(text)  # [B, T, E]
        transformer_out = self.transformer(embedded)  # [B, T, E]
        text_feat = transformer_out.mean(dim=1)  # Mean pooling over sequence

        if self.context_embedding is not None and context is not None:
            context_emb = self.context_embedding(context)     # [B, T, E]
            context_emb = torch.mean(context_emb, dim=1)      # [B, E]
            context_feat = self.relu(self.context_fc(context_emb))  # [B, H]
            combined = torch.cat((text_feat, context_feat), dim=1)
        else:
            combined = text_feat

        return self.fc(self.dropout(combined))

import torch
import torch.nn as nn


class IntentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(IntentNet, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim * 2]

        # Grab the last forward and first backward hidden states
        out_forward = lstm_out[:, -1, :self.lstm.hidden_size]
        out_backward = lstm_out[:, 0, self.lstm.hidden_size:]
        final_out = torch.cat((out_forward, out_backward), dim=1)  # [batch_size, hidden_dim * 2]

        logits = self.fc(final_out)  # [batch_size, output_dim]
        return logits

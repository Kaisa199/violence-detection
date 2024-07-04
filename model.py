import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size = 5000, num_classes = 2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 2)
        
    def forward(self, x):
        # Reshape x to (batchsize, 10 * 1000) = (batchsize, 10000)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = F.softmax(x, dim=1)
        return x

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Attention mechanism
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(hidden_size, 1)

        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize LSTM hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # lstm_out shape: (batch_size, seq_length, hidden_size)

        # Attention mechanism
        # Apply attention weights using a softmax activation

        attn_weights = F.softmax(self.W(lstm_out), dim=1)  # shape: (batch_size, seq_length, 1)
        
        # Apply attention weights to LSTM outputs
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # shape: (batch_size, 1, hidden_size)
        
        # Squeeze to remove unnecessary dimension
        attn_applied = attn_applied.squeeze(1)  # shape: (batch_size, hidden_size)
        attn_applied = self.output(attn_applied)
        
        return attn_applied
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project inputs to query, key, and value
        Q = self.query_projection(query)  # shape: (batch_size, seq_len, d_model)
        K = self.key_projection(key)      # shape: (batch_size, seq_len, d_model)
        V = self.value_projection(value)  # shape: (batch_size, seq_len, d_model)
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32).to(query.device))  # shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention weights to value projections
        attention_output = torch.matmul(attention_weights, V)  # shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape attention output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # shape: (batch_size, seq_len, d_model)
        
        # Apply output projection
        multihead_output = self.output_projection(attention_output)  # shape: (batch_size, seq_len, d_model)
        
        return multihead_output, attention_weights

class LSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(LSTMWithMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # MultiHeadAttention layer
        self.multihead_attention = MultiHeadAttention(hidden_size, num_heads)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize LSTM hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # lstm_out shape: (batch_size, seq_length, hidden_size)
        
        # Apply MultiHeadAttention on LSTM output
        multihead_out, attention_weights = self.multihead_attention(lstm_out, lstm_out, lstm_out)  # multihead_out shape: (batch_size, seq_length, hidden_size)
        multihead_out = multihead_out[:, -1, :]  # Consider only the last output of the sequence
        multihead_out = self.linear(multihead_out)
        return multihead_out, attention_weights
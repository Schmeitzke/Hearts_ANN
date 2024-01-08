import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ANN_Hearts(nn.Module):

    def __init__(self, input_size=260, hidden_size=128, output_size=52):
        super(ANN_Hearts, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

        self.card_vectors = np.zeros((52, 5), dtype=float)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation for first hidden layer
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # ReLU activation for second hidden layer
        x = self.dropout(x)  # Apply dropout
        x = F.softmax(self.output_layer(x), dim=0)  # Softmax for output layer
        return x

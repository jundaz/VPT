"""
Position feed-forward network from "Attention is All You Need"
"""

import torch.nn as nn
from c2nl.modules.util_class import LayerNorm
import torch.backends.cuda
import torch.backends.cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.intermediate = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.intermediate(self.layer_norm(x))))
        output = self.dropout_2(self.output(inter))
        return output + x


class CLSPositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff=768):
        super(CLSPositionwiseFeedForward, self).__init__()
        self.intermediate = nn.Linear(d_model, d_ff)
        self.layer_norm = LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
             [ batch_size, input_len, model_dim ]
        """
        return self.relu(self.intermediate(self.layer_norm(x)))
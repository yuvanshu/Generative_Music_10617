import torch
import torch.nn as nn
import numpy as np


def get_device():
    """
    Get the appropriate device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MusicRBM(nn.Module):
    def __init__(self, input_shape=(88, 64), num_hidden=128, device=None):
        """

        :param input_shape: the shape of input image
        :param num_hidden: the number of hidden nodes
        :param device: the device to run the model
        """
        super(MusicRBM, self).__init__()

        # Automatically select device
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # Model parameters
        self.input_shape = input_shape
        self.num_visible = np.prod(input_shape)
        self.num_hidden = num_hidden

        # initialize weights and biases
        self.weights = nn.Parameter(torch.randn(num_hidden, self.num_visible, device=self.device) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden, device=self.device))

        print(f"Model initialized on device: {self.device}")
        print(f"Input shape: {input_shape}, Visible nodes: {self.num_visible}, Hidden nodes: {num_hidden}")

    def forward(self, v):
        """forward pass from visible to hidden layer"""
        # Ensure input is flattened
        if len(v.shape) > 2:
            v = v.view(v.size(0), -1)

        hidden_activation = torch.matmul(v, self.weights.t()) + self.hidden_bias
        hidden_prob = torch.sigmoid(hidden_activation)
        return hidden_prob

    def sample_hidden(self, v):
        """sample hidden layer from visible layer"""
        # Ensure input is flattened
        if len(v.shape) > 2:
            v = v.view(v.size(0), -1)

        hidden_prob = self.forward(v)
        hidden_states = torch.bernoulli(hidden_prob)
        return hidden_prob, hidden_states

    def sample_visible(self, h):
        """sample visible layer from hidden layer"""
        visible_activation = torch.matmul(h, self.weights) + self.visible_bias
        visible_prob = torch.sigmoid(visible_activation)
        visible_states = torch.bernoulli(visible_prob)
        return visible_prob, visible_states

    def contrastive_divergence(self, noisy_v, clean_v, learning_rate=0.01, k=1):
        """
        Contrastive Divergence learning algorithm
        """
        # Ensure inputs are flattened
        noisy_v = noisy_v.view(noisy_v.size(0), -1)
        clean_v = clean_v.view(clean_v.size(0), -1)

        # Positive phase
        pos_hidden_prob, pos_hidden_states = self.sample_hidden(noisy_v)

        # Negative phase
        chain_v = noisy_v
        chain_hidden_states = pos_hidden_states

        for _ in range(k):
            # sample visible layer from hidden layer
            neg_visible_prob, neg_visible_states = self.sample_visible(chain_hidden_states)
            # sample hidden layer from visible layer
            chain_hidden_prob, chain_hidden_states = self.sample_hidden(neg_visible_prob)

        # Compute associations
        pos_association = torch.matmul(pos_hidden_prob.t(), noisy_v)
        neg_association = torch.matmul(chain_hidden_prob.t(), neg_visible_prob)

        # Update parameters
        self.weights.data += learning_rate * (pos_association - neg_association) / noisy_v.size(0)
        self.visible_bias.data += learning_rate * torch.mean(noisy_v - neg_visible_prob, dim=0)
        self.hidden_bias.data += learning_rate * torch.mean(pos_hidden_prob - chain_hidden_prob, dim=0)

        # Compute reconstruction error (MSE)
        reconstruction_error = torch.nn.functional.mse_loss(
            neg_visible_prob.view(-1, *self.input_shape),
            clean_v.view(-1, *self.input_shape)
        )
        return reconstruction_error

    def reconstruct(self, noisy_input, num_iterations=10):
        """
        Reconstruct clean input from noisy input using Gibbs sampling
        """
        # Ensure input is flattened
        if len(noisy_input.shape) > 2:
            noisy_input = noisy_input.view(noisy_input.size(0), -1)

        current_input = noisy_input.clone()

        for _ in range(num_iterations):
            # sample hidden layer from visible layer
            _, hidden_states = self.sample_hidden(current_input)
            # sample visible layer from hidden layer
            visible_prob, current_input = self.sample_visible(hidden_states)

        # Reshape to original shape
        return current_input.view(-1, *self.input_shape)
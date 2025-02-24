import torch
import torch.nn as nn
import torch.nn.functional as F


class FirConv(nn.Module):
    def __init__(self, fres=100, N=None, fs=250, weight=None):
        super().__init__()
        print("version 0.1.1")
        # Initialize trainable weight parameters
        if weight is None:
            # Default initialization: zero tensor with gradient tracking
            self.weight = nn.Parameter(torch.zeros(fres), requires_grad=True)
        else:
            # Validate provided weight dimensions
            if len(weight) != fres:
                raise ValueError(f"Provided weight length {len(weight)} does not match fres={fres}.")

            # Convert provided weights to Parameter with gradient tracking
            self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=True)

        self.weight_hard = None  # Stores binarized weights for analysis
        self.kernel = None       # Cached filter kernel

        # Configure filter length (ensure odd symmetry)
        self.N = N if N is not None else fs * 4
        if self.N % 2 == 0:
            self.N += 1

        self.nqy = fs * 0.5  # Nyquist frequency

        # Register non-trainable buffers
        self.register_buffer('hamming_window', torch.hamming_window(self.N))
        self.register_buffer('response', self.generate_response(fres))

    def generate_response(self, fres):
        """Generate frequency response matrix for FIR filter design."""
        alpha = 0.5 * (self.N - 1)  # Center point for symmetric filter
        m = (torch.arange(0, self.N) - alpha).view(1, -1)  # Time indices

        # Create overlapping frequency bands
        bands = torch.arange(1, fres + 2) / self.nqy  # Normalized frequencies
        left, right = bands[:-1].unsqueeze(1), bands[1:].unsqueeze(1)

        # Precompute sinc basis functions
        sinc_left = torch.sinc(left * m)  # Sinc for lower band edges
        sinc_right = torch.sinc(right * m)  # Sinc for upper band edges

        # Calculate band-limited responses using difference of sincs
        response = (right * sinc_right - left * sinc_left)

        return response

    def generate_kernel(self):
        """Generate FIR kernel using learned weights and windowing."""
        # Apply sigmoid for smooth [0,1] constraint
        weight = torch.sigmoid(self.weight)

        # Create hard thresholded version (straight-through estimator)
        weight_hard = torch.where(weight >= 0.5, torch.tensor(1.0, device=self.weight.device),
                                  torch.tensor(0, device=self.weight.device))
        weight = weight + (weight_hard - weight).detach()  # Bypass gradient

        self.weight_hard = weight_hard  # Store for visualization/analysis

        # Combine frequency responses using learned weights
        h = torch.matmul(weight, self.response)
        # Apply windowing and reshape for convolution
        kernel = (h * self.hamming_window).view(1, 1, 1, self.N)

        return kernel

    def forward(self, x):
        """Zero-phase filtering through forward-backward convolution."""
        # Kernel management: dynamic during training, cached during inference
        if self.training:
            self.kernel = self.generate_kernel()
        else:
            # Lazy initialization for inference
            if self.kernel is None:
                self.kernel = self.generate_kernel()

        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)
        
        # Forward-backward convolution for zero-phase response
        x = F.conv2d(x, self.kernel, padding=(0, (self.N - 1) // 2))
        x = torch.flip(x, dims=[-1])  # Reverse time axis
        x = F.conv2d(x, self.kernel, padding=(0, (self.N - 1) // 2))
        x = torch.flip(x, dims=[-1]).squeeze()  # Restore original time direction

        return x.squeeze()  # Remove singleton dimensions
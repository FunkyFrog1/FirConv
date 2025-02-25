# FirConv: A PyTorch-based module for building trainable FIR filter  [![Version](https://img.shields.io/badge/version-0.1.1-red.svg)](https://semver.org) [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-orange)](https://pytorch.org)

<img src="https://github.com/user-attachments/assets/923837e9-8192-466f-8985-6cdcd030f03a" width="600" alt="FirCNN Architecture Diagram">

## 🚀 Introduction
A PyTorch-based module for building **trainable FIR filters** in deep neural networks. Key features:

- 🔧 Seamless integration with existing PyTorch models
- 🎚️ Pre-defined filters or learnable coefficients
- 📊 Real-time frequency adaptation visualization
- ⚡ CUDA-accelerated computation

> "Enabling neural networks to understand signal processing through differentiable filters"

## 📦 Installation

### From Source
```bash
git clone https://github.com/FunkyFrog1/FirConv.git
cd FirConv
pip install .
```

### From PyPI (coming soon)
```bash
pip install firconv
```

## 🛠️ Basic Usage
```python
import torch
from firconv.firconv import FirConv

# Create learnable FIR filter
fir = FirConv(fres=100, fs=250)

# Process signal batch: (batch_size, channels, seq_len)
x = torch.randn(1, 63, 250)  # Batch of 63 channels signal
y = fir(x)  # Output shape: (1, 63, 250)
```

## 📚 Documentation
| Argument    | Type    | Default | Description                                         |
|-------------|---------|---------|-----------------------------------------------------|
| `fres`      | int     | 100     | Filter frequency range [0-fres]                     |
| `N`         | int     | -       | Filter window size(default compute by fres and fs)  |
| `fs`        | int     | 250     | Sample rate                                         |

## 📅 Release Notes
### 0.1.1 - 2025-02-24
- Initial public release
- Add CUDA kernel optimization
- Support multi-channel filtering

## 🤝 Contributing
We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📜 License
Apache 2.0 © 2025 FunkyFrog


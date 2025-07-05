# Federated Learning Simulation for Thesis Research

## What is This Project?

This is a federated learning simulation framework built for thesis research. It allows you to compare different federated learning algorithms (like FedAvg, FedProx, SCAFFOLD) against traditional centralized machine learning.

**Federated Learning** means training machine learning models across multiple devices/clients without sharing raw data - only model updates are shared. This preserves privacy while still allowing collaborative learning.

## What Can You Study With This?

This simulation helps answer important research questions:

- How does federated learning perform compared to centralized learning?
- What happens when clients have very different data (non-IID data)?
- Which federated learning algorithms work best in different scenarios?
- How much communication overhead do different algorithms create?
- How fair are the algorithms - do all clients benefit equally?

## Implemented Algorithms

- **FedAvg**: The basic federated averaging algorithm (baseline)
- **FedProx**: Handles clients with different data better than FedAvg
- **FedAdam**: Uses Adam optimizer on the server side
- **SCAFFOLD**: Reduces variance between client updates
- **COOP**: Cooperative learning approach

## Datasets You Can Use

- **MNIST**: Handwritten digits (0-9) - simple and fast for testing
- **CIFAR-10**: Small color images (10 categories) - more complex
- **Shakespeare**: Text data for language modeling
- **FEMNIST**: Federated version of handwritten characters

## How to Get Started

### What You Need First

- Python 3.8 or newer
- A computer with at least 4GB RAM
- About 2GB free disk space

### Step 1: Get the Code

**Option A: Download as ZIP**
1. Go to the GitHub repository
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to your desired location

**Option B: Clone with Git**
```bash
git clone https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower.git
cd Federated-Learning-Simulation-Flower
```

### Step 2: Install Requirements

Open a terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs all the necessary Python packages like PyTorch, NumPy, Pandas, etc.

### Step 3: Run Your First Experiment

For a quick test (takes about 5-10 minutes):

```bash
python enhanced_fl_thesis_organized.py
```

This will:
- Download the MNIST dataset automatically
- Run 2 federated learning rounds with 2 clients
- Compare FedAvg and FedProx algorithms
- Generate results and plots

## Understanding the Configuration

The `config.yaml` file controls everything. Key settings:

```yaml
# Which algorithms to test
algorithms:
  - "FedAvg"      # Basic federated averaging
  - "FedProx"     # Better for different client data

# Which datasets to use
datasets:
  - "mnist"       # Start with this - it's fast

# How many clients and rounds
num_clients: 2          # Number of simulated devices
num_rounds: 2           # Training rounds (increase for real experiments)
local_epochs: 2         # Local training on each client

# Data distribution
beta_values: [0.1]      # 0.1 = very different data per client
                        # 1.0 = similar data per client
```

## What Results Do You Get?

After running, check the `results/` folder. You'll find:

### CSV Files (for analysis)
- `federated_only_results.csv` - Main federated learning results
- `centralized_only_results.csv` - Centralized learning comparison
- `comprehensive_results.csv` - Everything combined

### Analysis Files
- `academic_analysis.json` - Statistical test results
- `research_analysis.json` - Research insights and conclusions

### Visualizations
- `academic_dashboards/` - HTML dashboards you can open in browser
- `thesis_plots/` - Publication-ready plots as PNG files

## Main Files Explained

- **`enhanced_fl_thesis_organized.py`** - The main experiment script that does everything
- **`config.yaml`** - Settings file where you control all parameters
- **`data_preparation.py`** - Handles downloading and preparing datasets
- **`generate_plots.py`** - Creates academic plots from your results
- **`requirements.txt`** - List of Python packages needed

## Running Different Experiments

### Quick Test (5 minutes)
Keep the default `config.yaml` settings - uses MNIST, 2 clients, 2 rounds.

### Medium Experiment (30 minutes)
Edit `config.yaml`:
```yaml
num_clients: 10
num_rounds: 20
datasets: ["mnist", "cifar10"]
```

### Full Thesis Experiment (several hours)
Edit `config.yaml`:
```yaml
algorithms: ["FedAvg", "FedProx", "SCAFFOLD", "FedAdam"]
datasets: ["mnist", "cifar10"]
num_clients: 20
num_rounds: 100
beta_values: [0.1, 0.5, 1.0]  # Test different data distributions
```

## Understanding the Data Distribution

The `beta_values` control how different the data is across clients:

- **beta = 0.1**: Very different data (extreme non-IID) - realistic for federated learning
- **beta = 0.5**: Moderately different data 
- **beta = 1.0**: Similar data across clients (closer to centralized learning)

Lower values make federated learning harder but more realistic.

## Troubleshooting

**"Out of memory" error**: Reduce `num_clients` or `batch_size` in config.yaml

**"No module named..." error**: Run `pip install -r requirements.txt` again

**Slow execution**: The first run downloads datasets. Subsequent runs are faster.

**Empty results**: Check that `num_rounds` > 0 and algorithms are spelled correctly in config.yaml

## For Thesis Writing

This simulation generates everything you need for academic writing:

- **Statistical analysis**: T-tests, confidence intervals, effect sizes
- **Publication-quality plots**: High-resolution figures for your thesis
- **Raw data**: CSV files for further analysis in Excel/R/SPSS
- **Ready-to-use insights**: JSON files with research conclusions

## What Makes This Different?

Unlike basic federated learning tutorials, this framework:

- Compares multiple algorithms systematically
- Handles realistic non-IID data distributions
- Provides statistical significance testing
- Generates thesis-ready visualizations
- Includes both federated and centralized baselines
- Measures communication costs and fairness metrics

## Getting Help

If something doesn't work:

1. Check that you've installed requirements: `pip install -r requirements.txt`
2. Make sure Python 3.8+ is installed: `python --version`
3. Try the default config first before making changes
4. Check the `results/` folder for error logs

## Author

**Mehdi MOUALIM**
- Thesis: Comparative Analysis of Federated Learning Algorithms
- GitHub: https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower

This project was created for master's thesis research in federated learning.

## GPU Configuration for Faster Simulations

By default, this simulation runs on CPU. If you have a GPU, you can significantly speed up training.

### Check if You Have GPU Support

First, check if PyTorch can see your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

### Enable GPU in Configuration

Edit `config.yaml` to use GPU:

```yaml
# System Configuration - GPU Setup
device: "cuda"              # Change from "cpu" to "cuda"
num_workers: 8              # Increase for GPU (match your CPU cores)
pin_memory: true            # Enable for GPU acceleration
```

### GPU Performance Settings

For optimal GPU performance, also adjust these settings:

```yaml
# Increase batch sizes for GPU
batch_size: 64              # Increase from 32 to 64 or 128

# Increase clients for parallel processing
num_clients: 10             # More clients can run simultaneously
clients_per_round: 8        # Increase participation

# Use larger models that benefit from GPU
model_architectures:
  cifar10:
    - model_type: "resnet18"  # GPU handles larger models well
    - model_type: "resnet34"  # Even larger for better performance
```

### Force Specific GPU (Multi-GPU Systems)

If you have multiple GPUs, specify which one to use:

```yaml
device: "cuda:0"            # Use first GPU
# device: "cuda:1"          # Use second GPU
```

Or set it in your terminal before running:

```bash
# Windows
set CUDA_VISIBLE_DEVICES=0
python enhanced_fl_thesis_organized.py

# Linux/Mac
CUDA_VISIBLE_DEVICES=0 python enhanced_fl_thesis_organized.py
```

### Expected Speed Improvements

With GPU, you can expect:

- **2-5x faster** for CNN models on MNIST/CIFAR-10
- **5-10x faster** for ResNet models
- **3-8x faster** for LSTM models on text data

### GPU Memory Management

If you get "out of memory" errors:

```yaml
# Reduce batch size
batch_size: 32              # Start here, reduce if needed

# Reduce number of clients
num_clients: 5              # Fewer clients = less memory

# Use gradient checkpointing (add to config.yaml)
model_settings:
  gradient_checkpointing: true
```

### Verify GPU Usage During Training

While running, check GPU usage:

```bash
# Windows (if you have nvidia-smi)
nvidia-smi

# Check in Python during training
import torch
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
```

### Mixed Precision for Extra Speed

For even faster training on modern GPUs, enable mixed precision in `config.yaml`:

```yaml
# Training optimizations
mixed_precision: true       # Uses less memory, runs faster
```

**Note**: The current default configuration uses CPU because it works on all systems. Change to GPU settings only if you have a compatible NVIDIA GPU with CUDA support.

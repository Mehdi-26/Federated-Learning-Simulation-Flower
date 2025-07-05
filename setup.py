#!/usr/bin/env python3
"""
Automated Setup Script for Federated Learning Thesis Project
===========================================================
This script automatically creates the necessary directory structure
and sets up the environment for your FL thesis project.

Run this script first before running your experiments.

Author: Mehdi MOUALIM
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def create_directory_structure():
    """Create the necessary directory structure for the project"""
    
    # Define the directory structure
    directories = [
        "data",
        "data/mnist",
        "data/cifar10", 
        "data/femnist",
        "results",
        "results/plots",
        "results/logs",
        "results/models",
        "results/academic_dashboards",
        "results/statistical_analysis",
        "notebooks",
        "docs",
        "experiments"
    ]
    
    print("🏗️  Creating directory structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    print("📁 Directory structure created successfully!")

def create_data_info_file():
    """Create a data information file"""
    
    data_info = {
        "datasets": {
            "mnist": {
                "description": "MNIST handwritten digits dataset",
                "classes": 10,
                "input_shape": [1, 28, 28],
                "samples": {"train": 60000, "test": 10000},
                "auto_download": True
            },
            "cifar10": {
                "description": "CIFAR-10 object recognition dataset", 
                "classes": 10,
                "input_shape": [3, 32, 32],
                "samples": {"train": 50000, "test": 10000},
                "auto_download": True
            },
            "femnist": {
                "description": "Federated EMNIST character recognition",
                "classes": 26,
                "input_shape": [1, 28, 28], 
                "samples": {"train": "variable", "test": "variable"},
                "auto_download": True,
                "note": "EMNIST letters split used as proxy for FEMNIST"
            }
        },
        "data_distributions": {
            "iid": "Independent and identically distributed - balanced across clients",
            "non_iid": "Non-IID using Dirichlet distribution with different beta values"
        },
        "heterogeneity_levels": {
            "beta_0.1": "High heterogeneity - very skewed data distribution",
            "beta_0.5": "Medium heterogeneity - moderately skewed distribution", 
            "beta_1.0": "Low heterogeneity - relatively balanced distribution"
        }
    }
    
    with open("data/dataset_info.json", "w") as f:
        json.dump(data_info, f, indent=2)
    
    print("📊 Dataset information file created!")

def create_gitignore():
    """Create a .gitignore file for the project"""
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPY-PKG-INFO

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VSCode
.vscode/

# Data files (too large for git)
data/mnist/
data/cifar10/
data/femnist/
*.pt
*.pth

# Result files (optional - you might want to track some results)
results/enhanced_thesis_*/
results/plots/*.png
results/models/
*.log

# System files
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
~*

# Large experimental files
experiments/large_experiments/
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("🚫 .gitignore file created!")

def create_readme():
    """Create a comprehensive README file (only if it doesn't exist)"""
    
    # Check if README already exists
    if Path("README.md").exists():
        print("📖 README.md already exists - skipping creation")
        return
    
    readme_content = """# 🎓 Federated Learning Thesis - Comprehensive Analysis

## Overview
This project implements a comprehensive federated learning analysis framework for academic research, specifically designed for master's thesis work on federated learning algorithms.

## 🔬 Research Questions Addressed

1. **FL vs Centralized Learning**: How does federated learning compare to traditional centralized machine learning?
2. **Non-IID Data Impact**: How does data heterogeneity affect federated learning performance?
3. **Device Reliability**: Can federated learning perform well with irregular device participation?
4. **Communication Efficiency**: What are the trade-offs between communication cost and accuracy?
5. **Algorithm Fairness**: Which algorithms provide the most equitable performance across clients?

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project structure
python setup.py
```

### 2. Configure Experiment
Edit `config.yaml` to specify:
- Algorithms to test
- Datasets to use
- Experimental parameters

### 3. Run Experiments
```bash
python enhanced_fl_thesis.py
```

### 4. View Results
Results will be saved in `results/enhanced_thesis_*/`:
- Interactive dashboards (HTML files)
- Statistical analysis (CSV files)
- Publication-ready plots (PNG files)

## 📊 Algorithms Supported

- **FedAvg**: Federated Averaging (baseline)
- **FedProx**: Federated Optimization with proximal term
- **FedAdam**: Federated Adam optimization
- **SCAFFOLD**: Stochastic Controlled Averaging
- **COOP**: Cooperative federated learning

## 📚 Datasets Supported

- **MNIST**: Handwritten digits (28x28, 10 classes)
- **CIFAR-10**: Object recognition (32x32, 10 classes)
- **FEMNIST**: Federated EMNIST characters (28x28, 26 classes)

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
# Basic setup
algorithms: ["FedAvg", "FedProx"]
datasets: ["mnist"]
num_clients: 10
num_rounds: 50

# Data heterogeneity
data_distributions: ["iid", "non_iid"]
beta_values: [0.1, 0.5, 1.0]  # Lower = more heterogeneous

# Academic analysis
research_questions:
  fl_vs_centralized: true
  non_iid_impact: true
  device_reliability: true
```

## 📈 Output Files

### Academic Dashboards
- `research_questions_dashboard.png` - Main research findings
- `statistical_analysis_dashboard.png` - Statistical tests and significance
- `interactive_academic_dashboard.html` - Interactive exploration

### Data Files
- `comprehensive_results.csv` - All metrics for statistical analysis
- `federated_results.json` - Detailed FL experiment results
- `academic_analysis.json` - Statistical test results

## 🎯 Using Results in Your Thesis

The framework generates publication-ready:
- **Figures**: High-resolution plots for thesis document
- **Tables**: Statistical results with p-values and effect sizes
- **Data**: Raw results for custom analysis in R/SPSS
- **Interactive Visualizations**: For thesis defense presentations

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{moualim2024federated,
  title={Comparative Analysis of Federated Learning Algorithms: Performance, Efficiency, and Fairness},
  author={Moualim, Mehdi},
  year={2024},
  school={Your University}
}
```

## 🤝 Contributing

This is an academic research project. Contributions and suggestions are welcome for:
- Additional FL algorithms
- New evaluation metrics
- Enhanced statistical analysis
- Additional datasets

## 📞 Contact

For questions about this research or collaboration:
- Author: Mehdi MOUALIM
- Email: [Your Email]
- Institution: [Your University]

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
"""

    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("📖 README.md file created!")

def install_dependencies():
    """Install Python dependencies"""
    
    print("📦 Installing Python dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            print("❌ requirements.txt not found! Please create it first.")
            return False
            
        # Install dependencies
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_system_requirements():
    """Check if system meets requirements"""
    
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available space
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)  # GB
    if free_space < 2:
        print(f"⚠️  Low disk space: {free_space:.1f}GB (recommend 2GB+)")
    else:
        print(f"✅ Disk space: {free_space:.1f}GB available")
    
    # Check if torch is available
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True

def create_sample_experiment():
    """Create a sample experiment configuration"""
    
    sample_config = """# Sample Quick Experiment Configuration
# This is a minimal setup for testing the framework

experiment_name: "quick_test_experiment"
description: "Quick test of FL framework"
author: "Mehdi MOUALIM"
thesis_title: "Comparative Analysis of Federated Learning Algorithms"

# System Configuration
random_seed: 42
device: "cpu"                   # Change to "cuda" if you have GPU
num_workers: 4
pin_memory: false

# Minimal algorithm set for quick testing
algorithms:
  - "FedAvg"
  - "FedProx"

# Algorithm parameters
algorithm_params:
  fedavg: {}
  fedprox:
    mu: 0.01

# Single dataset for quick testing  
datasets:
  - "mnist"

# Model architectures
model_architectures:
  mnist:
    - model_type: "cnn"

# Data heterogeneity (quick test)
beta_values: [0.1, 1.0]  # Test extreme and balanced distributions

# Basic federation settings
num_clients: 5
clients_per_round: 3
num_rounds: 10
local_epochs: 2

# Training settings
learning_rate: 0.01
batch_size: 32
weight_decay: 1e-4

# Academic analysis
academic_analysis:
  statistical_testing: true
  confidence_level: 0.95

# Research questions
research_questions:
  fl_vs_centralized: true
  non_iid_impact: true
  device_reliability: true
  communication_efficiency: true
  privacy_vs_accuracy: true
  scalability_analysis: true

# Privacy settings (disabled for quick test)
differential_privacy: false
"""

    with open("experiments/sample_quick_test.yaml", "w") as f:
        f.write(sample_config)
    
    print("🧪 Sample experiment configuration created!")

def main():
    """Main setup function"""
    
    print("🎓 Federated Learning Thesis Project Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met!")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Create supporting files
    create_data_info_file()
    create_gitignore()
    create_readme()
    create_sample_experiment()
    
    # Install dependencies
    install_success = install_dependencies()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Review and customize config.yaml")
    print("2. Run: python enhanced_fl_thesis_organized.py")
    print("3. Check results in results/ directory")
    print("4. Generate plots: python generate_thesis_plots.py")
    
    if not install_success:
        print("\n⚠️  Note: Manual dependency installation may be required")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

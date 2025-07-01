# 🎓 Federated Learning Simulation Framework for Master's Thesis

## 📋 Overview

This repository contains a comprehensive federated learning simulation framework designed for academic research, specifically for master's thesis work analyzing federated learning algorithms. The framework implements multiple state-of-the-art FL algorithms and provides extensive academic visualizations and analysis tools.

## 🔬 Research Questions Addressed

1. **FL vs Centralized Learning**: How does federated learning compare to traditional centralized machine learning in terms of data security and model accuracy?
2. **Non-IID Data Impact**: How does data heterogeneity affect federated learning performance across different algorithms?
3. **Device Reliability**: Can federated learning maintain performance when devices have irregular participation?
4. **Communication Efficiency**: What are the trade-offs between communication cost and model accuracy?
5. **Algorithm Fairness**: Which algorithms provide the most equitable performance across diverse clients?

## 🧠 Implemented Algorithms

- **FedAvg**: Federated Averaging (baseline)
- **FedProx**: Federated Optimization with proximal term
- **FedAdam**: Federated Adam optimization
- **SCAFFOLD**: Stochastic Controlled Averaging for Federated Learning
- **COOP**: Cooperative Federated Learning

## 📊 Datasets Supported

- **MNIST**: Handwritten digits (28x28, 10 classes)
- **CIFAR-10**: Object recognition (32x32, 10 classes)
- **FEMNIST**: Federated EMNIST characters (28x28, 26 classes)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower.git
cd Federated-Learning-Simulation-Flower
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup the project**:
```bash
python setup.py
```

### Running Experiments

#### Quick Test (5-10 minutes)
```bash
python enhanced_fl_thesis_organized.py
```

#### Full Academic Analysis
Modify `config.yaml` for comprehensive experiments:
```yaml
algorithms: ["FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "COOP"]
datasets: ["mnist", "cifar10"]
num_rounds: 100
num_clients: 10
data_distributions: ["iid", "non_iid"]
beta_values: [0.1, 0.5, 1.0]
```

## 📈 Generated Outputs

### Academic Dashboards
- `training_progress_dashboard.png` - Training curves and convergence analysis
- `algorithm_summary_dashboard.png` - Performance comparison with summary table
- `performance_metrics_dashboard.png` - Key performance metrics visualization
- `communication_fairness_dashboard.png` - Communication efficiency and fairness analysis
- `research_questions_dashboard.png` - Direct research question answers

### Data Files
- `comprehensive_results.csv` - All metrics for statistical analysis
- `federated_results.json` - Detailed FL experiment results
- `centralized_results.json` - Baseline comparison results
- `academic_analysis.json` - Statistical significance tests
- `research_analysis.json` - Research question specific analysis

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
# Algorithms to compare
algorithms: ["FedAvg", "FedProx", "SCAFFOLD"]

# Datasets for evaluation
datasets: ["mnist"]

# Federation parameters
num_clients: 10
clients_per_round: 6
num_rounds: 50
local_epochs: 3

# Data heterogeneity
data_distributions: ["iid", "non_iid"]
beta_values: [0.1, 0.5, 1.0]  # Lower = more heterogeneous

# Robustness testing
robustness_testing:
  dropout_simulation: true
  dropout_rates: [0.0, 0.1, 0.2, 0.3]
```

## 📊 Academic Features

### Statistical Analysis
- **Significance Testing**: T-tests and ANOVA for algorithm comparison
- **Effect Size Analysis**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all reported metrics
- **Power Analysis**: Sample size validation

### Visualization Types
- **Training Curves**: Accuracy and loss evolution
- **Performance Comparisons**: Bar charts and box plots
- **Communication Analysis**: Efficiency scatter plots
- **Fairness Metrics**: Gini coefficient tracking
- **Heterogeneity Impact**: Non-IID degradation analysis

### Research Metrics
- **Model Drift**: Parameter divergence between clients
- **Client Fairness**: Gini-based equity measurement
- **Consensus Score**: Agreement between clients and global model
- **Gradient Diversity**: Cosine similarity of client gradients
- **Participation Rate**: Client availability tracking

## 📁 Project Structure

```
federated-learning-simulation/
├── enhanced_fl_thesis_organized.py    # Main simulation framework
├── config.yaml                        # Experiment configuration
├── requirements.txt                   # Python dependencies
├── setup.py                          # Project setup script
├── data_preparation.py               # Dataset handling
├── run_complete_pipeline.py          # Automated execution
├── data/                             # Dataset storage
│   ├── mnist/                       # MNIST dataset
│   └── dataset_info.json           # Dataset metadata
├── results/                          # Experiment results
│   └── enhanced_thesis_*/           # Timestamped results
│       ├── academic_dashboards/     # Publication-ready plots
│       ├── comprehensive_results.csv
│       └── *.json                   # Analysis results
└── notebooks/                       # Jupyter analysis notebooks
```

## 🎯 For Thesis Writing

The framework generates publication-ready materials:

- **Figures**: High-resolution plots for thesis document
- **Tables**: Statistical results with p-values and effect sizes
- **Data**: Raw results for custom analysis in R/SPSS
- **Interactive Visualizations**: For thesis defense presentations

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{moualim2024federated,
  title={Comparative Analysis of Federated Learning Algorithms: Performance, Efficiency, and Fairness},
  author={Moualim, Mehdi},
  year={2024},
  school={Your University},
  url={https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request for:
- Additional FL algorithms
- New evaluation metrics
- Enhanced statistical analysis
- Additional datasets
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Mehdi MOUALIM
- **GitHub**: [@Mehdi-26](https://github.com/Mehdi-26)
- **Project**: [Federated Learning Simulation Framework](https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower)

## 🙏 Acknowledgments

- Flower Framework for federated learning algorithms
- PyTorch team for the deep learning framework
- Academic community for federated learning research

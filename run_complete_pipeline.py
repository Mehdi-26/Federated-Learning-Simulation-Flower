#!/usr/bin/env python3
"""
Complete Execution Script for Federated Learning Thesis
======================================================
This script orchestrates the complete experimental pipeline:
1. Environment setup and validation
2. Data preparation and validation  
3. Federated learning experiments
4. Results analysis and visualization
5. Report generation

Author: Mehdi MOUALIM
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import yaml
import json

def print_header(title, char="="):
    """Print a formatted header"""
    print(f"\n{char * 60}")
    print(f"🎓 {title}")
    print(f"{char * 60}")

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {title}")
    print("-" * 40)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_step(1, "Checking Prerequisites")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        print("✅ Python version OK")
    
    # Check required files
    required_files = [
        "config.yaml",
        "enhanced_fl_thesis.py", 
        "requirements.txt"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            issues.append(f"Missing file: {file}")
        else:
            print(f"✅ Found: {file}")
    
    # Check data directory
    if not Path("data").exists():
        print("⚠️  Data directory not found - will be created")
    else:
        print("✅ Data directory exists")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU")
    except ImportError:
        issues.append("PyTorch not installed")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("\n✅ All prerequisites met!")
    return True

def setup_environment():
    """Setup the project environment"""
    print_step(2, "Setting Up Environment")
    
    try:
        # Create directories if they don't exist
        directories = ["data", "results", "experiments", "notebooks"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Directory ready: {directory}/")
        
        # Install dependencies if needed
        try:
            import torch, torchvision, numpy, pandas, matplotlib, seaborn, plotly, yaml, scipy, sklearn
            print("✅ All dependencies available")
        except ImportError as e:
            print(f"⚠️  Installing missing dependencies...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
            else:
                print(f"❌ Error installing dependencies: {result.stderr}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def prepare_data():
    """Prepare and validate datasets"""
    print_step(3, "Preparing Datasets")
    
    try:
        # Run data preparation script
        if Path("data_preparation.py").exists():
            print("🔄 Running data preparation script...")
            result = subprocess.run([sys.executable, "data_preparation.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Data preparation completed")
                print(result.stdout)
            else:
                print(f"⚠️  Data preparation script failed: {result.stderr}")
        
        # Alternative: Download datasets directly
        print("🔄 Verifying datasets...")
        import torch
        from torchvision import datasets, transforms
        
        data_dir = Path("data")
        
        # Test MNIST download
        try:
            datasets.MNIST(root=data_dir / 'mnist', train=True, download=True)
            print("✅ MNIST ready")
        except Exception as e:
            print(f"⚠️  MNIST issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False

def load_configuration():
    """Load and validate experiment configuration"""
    print_step(4, "Loading Configuration")
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded")
        print(f"   📊 Algorithms: {config.get('algorithms', [])}")
        print(f"   📚 Datasets: {config.get('datasets', [])}")
        print(f"   🔄 Rounds: {config.get('num_rounds', 'Not specified')}")
        print(f"   👥 Clients: {config.get('num_clients', 'Not specified')}")
        
        # Validate configuration
        required_keys = ['algorithms', 'datasets', 'num_rounds', 'num_clients']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"⚠️  Missing configuration keys: {missing_keys}")
            return None
        
        return config
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return None

def run_experiments(config):
    """Run the federated learning experiments"""
    print_step(5, "Running Federated Learning Experiments")
    
    # Calculate estimated time
    num_algorithms = len(config['algorithms'])
    num_datasets = len(config['datasets'])
    num_rounds = config['num_rounds']
    
    # Estimate time (rough calculation)
    estimated_minutes = num_algorithms * num_datasets * num_rounds * 0.1  # ~6 seconds per round
    
    print(f"📊 Experiment Overview:")
    print(f"   🧠 Algorithms: {num_algorithms}")
    print(f"   📚 Datasets: {num_datasets}")
    print(f"   🔄 Rounds per experiment: {num_rounds}")
    print(f"   ⏱️  Estimated time: {estimated_minutes:.1f} minutes")
    
    # Ask for confirmation
    response = input(f"\n🤔 Proceed with experiments? (y/n): ").lower()
    if response != 'y':
        print("❌ Experiments cancelled by user")
        return False
    
    try:
        print("\n🚀 Starting experiments...")
        start_time = time.time()
        
        # Import and run the enhanced FL framework
        sys.path.insert(0, '.')
        
        # Run the main enhanced FL simulation
        if Path("enhanced_fl_thesis.py").exists():
            result = subprocess.run([sys.executable, "enhanced_fl_thesis.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                end_time = time.time()
                actual_minutes = (end_time - start_time) / 60
                
                print(f"✅ Experiments completed successfully!")
                print(f"⏱️  Actual time: {actual_minutes:.1f} minutes")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return True
            else:
                print(f"❌ Experiments failed: {result.stderr}")
                return False
        else:
            print("❌ Enhanced FL thesis script not found!")
            return False
            
    except Exception as e:
        print(f"❌ Experiment execution failed: {e}")
        return False

def analyze_results():
    """Analyze and summarize experimental results"""
    print_step(6, "Analyzing Results")
    
    try:
        # Find the most recent results directory
        results_dirs = list(Path("results").glob("enhanced_thesis_*"))
        if not results_dirs:
            print("❌ No results directories found")
            return False
        
        latest_results_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        print(f"📁 Latest results: {latest_results_dir}")
        
        # Check for key result files
        key_files = [
            "comprehensive_results.csv",
            "academic_analysis.json",
            "research_analysis.json"
        ]
        
        found_files = []
        for file in key_files:
            file_path = latest_results_dir / file
            if file_path.exists():
                found_files.append(file)
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"✅ {file} ({file_size:.1f} KB)")
            else:
                print(f"⚠️  Missing: {file}")
        
        # Check for dashboards
        dashboard_dir = latest_results_dir / "academic_dashboards"
        if dashboard_dir.exists():
            dashboard_files = list(dashboard_dir.glob("*.png")) + list(dashboard_dir.glob("*.html"))
            print(f"📊 Generated {len(dashboard_files)} dashboard files")
            for file in dashboard_files:
                print(f"   📈 {file.name}")
        
        # Quick results summary
        try:
            csv_file = latest_results_dir / "comprehensive_results.csv"
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                print(f"\n📊 Quick Results Summary:")
                print(f"   📈 Total experiments: {len(df)}")
                print(f"   🎯 Best accuracy: {df['final_accuracy'].max():.3f}")
                print(f"   📉 Worst accuracy: {df['final_accuracy'].min():.3f}")
                print(f"   📊 Average accuracy: {df['final_accuracy'].mean():.3f}")
                
                if 'algorithm' in df.columns:
                    best_algorithm = df.loc[df['final_accuracy'].idxmax(), 'algorithm']
                    print(f"   🏆 Best algorithm: {best_algorithm}")
        
        except Exception as e:
            print(f"⚠️  Could not analyze CSV results: {e}")
        
        return len(found_files) > 0
        
    except Exception as e:
        print(f"❌ Results analysis failed: {e}")
        return False

def generate_report():
    """Generate a final summary report"""
    print_step(7, "Generating Final Report")
    
    try:
        # Find latest results
        results_dirs = list(Path("results").glob("enhanced_thesis_*"))
        if not results_dirs:
            print("❌ No results to report")
            return False
        
        latest_results_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        
        # Generate summary report
        report_content = f"""# Federated Learning Thesis - Experiment Report

## Experiment Overview
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Results Directory**: {latest_results_dir}

## Files Generated
"""
        
        # List all generated files
        for file_path in latest_results_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(latest_results_dir)
                file_size = file_path.stat().st_size
                if file_size > 1024*1024:  # MB
                    size_str = f"{file_size/(1024*1024):.1f} MB"
                elif file_size > 1024:  # KB
                    size_str = f"{file_size/1024:.1f} KB"
                else:
                    size_str = f"{file_size} bytes"
                
                report_content += f"- `{relative_path}` ({size_str})\n"
        
        report_content += f"""
## Next Steps for Thesis

### 1. Review Results
- Open `academic_dashboards/interactive_academic_dashboard.html` in your browser
- Examine `research_questions_dashboard.png` for main findings
- Review `comprehensive_results.csv` for detailed analysis

### 2. Statistical Analysis
- Import `comprehensive_results.csv` into R/SPSS for advanced statistics
- Use `academic_analysis.json` for significance test results
- Check `statistical_analysis_dashboard.png` for p-values and effect sizes

### 3. Thesis Writing
- Use publication-ready plots from `academic_dashboards/`
- Reference statistical findings from analysis files
- Incorporate research question answers from dashboards

### 4. Presentation Preparation
- Use interactive dashboard for thesis defense
- Extract key figures for presentation slides
- Prepare talking points from research analysis

## Research Questions Addressed
1. **FL vs Centralized Learning**: Performance comparison with privacy trade-offs
2. **Non-IID Data Impact**: Heterogeneity effects on algorithm performance  
3. **Device Reliability**: Robustness to irregular participation
4. **Communication Efficiency**: Cost vs accuracy trade-offs
5. **Algorithm Fairness**: Equity analysis across diverse clients

## Contact
For questions about this analysis framework:
- Author: Mehdi MOUALIM
- Framework: Enhanced Academic FL Analysis
"""
        
        # Save report
        report_path = latest_results_dir / "EXPERIMENT_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Report generated: {report_path}")
        print(f"📊 Results directory: {latest_results_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def main():
    """Main execution pipeline"""
    
    print_header("Federated Learning Thesis - Complete Execution Pipeline")
    
    # Execute pipeline steps
    steps = [
        ("Prerequisites Check", check_prerequisites),
        ("Environment Setup", setup_environment), 
        ("Data Preparation", prepare_data),
        ("Configuration Loading", load_configuration),
    ]
    
    config = None
    
    # Execute initial steps
    for step_name, step_func in steps:
        if step_name == "Configuration Loading":
            config = step_func()
            if config is None:
                print("❌ Pipeline failed at configuration loading")
                return
        else:
            success = step_func()
            if not success:
                print(f"❌ Pipeline failed at: {step_name}")
                return
    
    # Run experiments
    success = run_experiments(config)
    if not success:
        print("❌ Pipeline failed at experiments")
        return
    
    # Analyze results
    success = analyze_results()
    if not success:
        print("❌ Pipeline failed at results analysis")
        return
    
    # Generate final report
    success = generate_report()
    
    # Final summary
    print_header("Execution Complete!", "🎉")
    
    if success:
        print("✅ All steps completed successfully!")
        print("\n📋 What to do next:")
        print("1. 📊 Open interactive dashboard in your browser")
        print("2. 📈 Review academic analysis results") 
        print("3. 📝 Use findings in your thesis")
        print("4. 🎓 Prepare for thesis defense")
        
        # Try to open interactive dashboard
        results_dirs = list(Path("results").glob("enhanced_thesis_*"))
        if results_dirs:
            latest_results_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
            dashboard_path = latest_results_dir / "academic_dashboards" / "interactive_academic_dashboard.html"
            
            if dashboard_path.exists():
                print(f"\n🌐 Interactive dashboard: {dashboard_path}")
                try:
                    import webbrowser
                    webbrowser.open(f'file://{dashboard_path.absolute()}')
                    print("🚀 Opening dashboard in browser...")
                except:
                    print("💡 Manually open the dashboard file in your browser")
    else:
        print("⚠️  Some steps had issues, but basic execution completed")
        print("📋 Check the results directory for partial outputs")

if __name__ == "__main__":
    main()
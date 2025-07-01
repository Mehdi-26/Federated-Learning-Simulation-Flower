esults.json                   0,01 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
t10k-labels-idx1-ubyte                   0,01 C:\Users\AICHA\OneDrive\Documents\Experiment 2\data\mnist\raw\t10...
federated_results.json                   0,01 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
t10k-labels-idx1-ubyte.gz                   0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\data\mnist\raw\t10... 
config.yaml                                 0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\config.yaml
experiment_config.yaml                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
experiment_config.yaml                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
dataset_info.json                           0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\data\dataset_info.... 
comprehensive_results.csv                   0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
comprehensive_results.csv                   0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
research_analysis.json                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
federated_only_results.csv                  0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
federated_only_results.csv                  0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
research_analysis.json                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
.gitignore                                  0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\.gitignore
academic_analysis.json                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t...
requirements.txt                            0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\requirements.txt      
centralized_results.json                    0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
academic_analysis.json                      0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
centralized_results.json                    0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
centralized_only_results.csv                0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
centralized_only_results.csv                0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\results\enhanced_t... 
README.md                                   0 C:\Users\AICHA\OneDrive\Documents\Experiment 2\README.md


PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git init
Initialized empty Git repository in C:/Users/AICHA/OneDrive/Documents/Experiment 2/.git/
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git lfs install
Updated Git hooks.
Git LFS initialized.
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git remote add origin https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower.git
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add .
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   .gitattributes
        new file:   .gitignore
        new file:   README.md
        new file:   config.yaml
        new file:   data/dataset_info.json
        new file:   data_preparation.py
        new file:   enhanced_fl_thesis_organized.py
        new file:   requirements.txt
        new file:   run_complete_pipeline.py
        new file:   setup.py

PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git commit -m "Initial commit: Enhanced Federated Learning Simulation Framework
>>
>> - Complete federated learning simulation with FedAvg and FedProx algorithms
>> - Academic research dashboard generation with comprehensive visualizations
>> - Enhanced data preprocessing and heterogeneity analysis
>> - Centralized vs federated learning comparison
>> - Non-IID data distribution analysis with Dirichlet sampling
>> - Comprehensive CSV and JSON result outputs for statistical analysis
>> - BatchNorm state dict handling for proper model aggregation
>> - Multi-objective algorithm evaluation framework
>>
>> Features:
>> - Adaptive neural network models for MNIST and CIFAR-10
>> - Real-time training metrics and convergence analysis
>> - Client fairness and participation tracking
>> - Communication cost optimization
>> - Academic thesis-ready visualizations and reports"
[master (root-commit) ca8e51b] Initial commit: Enhanced Federated Learning Simulation Framework
 10 files changed, 2962 insertions(+)
 create mode 100644 .gitattributes
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 config.yaml
 create mode 100644 data/dataset_info.json
 create mode 100644 data_preparation.py
 create mode 100644 enhanced_fl_thesis_organized.py
 create mode 100644 requirements.txt
 create mode 100644 run_complete_pipeline.py
 create mode 100644 setup.py
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add data/mnist/
The following paths are ignored by one of your .gitignore files:
data/mnist
hint: Use -f if you really want to add them.
hint: Disable this message with "git config set advice.addIgnoredFile false"
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add data/mnist/
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add results/
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add .gitignore
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git commit -m "Add datasets and experimental results
>>
>> - MNIST dataset files (handled with Git LFS)
>> - Comprehensive experimental results with academic dashboards
>> - Enhanced research visualizations and analysis
>> - CSV and JSON outputs for statistical analysis
>> - Multi-panel academic dashboard images
>>
>> Large files managed with Git LFS:
>> - MNIST raw dataset files
>> - Academic dashboard PNG files
>> - Research analysis visualizations"
[master 2301870] Add datasets and experimental results
 8 files changed, 24 insertions(+)
 create mode 100644 data/mnist/raw/t10k-images-idx3-ubyte
 create mode 100644 data/mnist/raw/t10k-images-idx3-ubyte.gz
 create mode 100644 data/mnist/raw/t10k-labels-idx1-ubyte
 create mode 100644 data/mnist/raw/t10k-labels-idx1-ubyte.gz
 create mode 100644 data/mnist/raw/train-images-idx3-ubyte
 create mode 100644 data/mnist/raw/train-images-idx3-ubyte.gz
 create mode 100644 data/mnist/raw/train-labels-idx1-ubyte
 create mode 100644 data/mnist/raw/train-labels-idx1-ubyte.gz
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add .
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git commit -m "Add comprehensive experimental results and analysis
>>
>> - Complete federated learning experimental results
>> - Academic research dashboards and visualizations  
>> - Statistical analysis outputs (CSV and JSON)
>> - Algorithm performance comparisons
>> - Research question analysis results
>> - Multi-objective evaluation metrics"
On branch master
nothing to commit, working tree clean
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git push -u origin master
Uploading LFS objects: 100% (8/8), 66 MB | 108 KB/s, done.
Enumerating objects: 26, done.
Counting objects: 100% (26/26), done.
Delta compression using up to 16 threads
Compressing objects: 100% (23/23), done.
Writing objects: 100% (26/26), 111.21 KiB | 15.89 MiB/s, done.
Total 26 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), done.
remote:
remote: Create a pull request for 'master' on GitHub by visiting:
remote:      https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower/pull/new/master
remote:
To https://github.com/Mehdi-26/Federated-Learning-Simulation-Flower.git
 * [new branch]      master -> master
branch 'master' set up to track 'origin/master'.
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> python -c "import enhanced_fl_thesis_organized; print(' Script imports successfully')"
 Script imports successfully
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add enhanced_fl_thesis_organized.py && git commit -m "Fix: Removed duplicate imports and completed missing code sections"
Au caractère Ligne:1 : 41
+ git add enhanced_fl_thesis_organized.py && git commit -m "Fix: Remove ...
+                                         ~~
Le jeton « && » n’est pas un séparateur d’instruction valide.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : InvalidEndOfLine

PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git add enhanced_fl_thesis_organized.py
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> git commit -m "Fix: Removed duplicate imports and completed missing code sections"
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
PS C:\Users\AICHA\OneDrive\Documents\Experiment 2> 

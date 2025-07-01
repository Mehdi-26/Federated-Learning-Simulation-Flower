#!/usr/bin/env python3
"""
Data Preparation Script for Federated Learning Thesis
=====================================================
This script handles data preparation, validation, and setup
for federated learning experiments.

Author: Mehdi MOUALIM
"""

import os
import json
import datetime
import torch
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class DatasetManager:
    """Manages dataset downloading, validation, and preparation"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_all_datasets(self):
        """Download all supported datasets"""
        print("📥 Downloading datasets...")
        
        # MNIST
        print("   🔢 Downloading MNIST...")
        self._download_mnist()
        
        # CIFAR-10
        print("   🖼️  Downloading CIFAR-10...")
        self._download_cifar10()
        
        # FEMNIST (using EMNIST as proxy)
        print("   ✍️  Downloading EMNIST (FEMNIST proxy)...")
        self._download_femnist()
        
        print("✅ All datasets downloaded!")
    
    def _download_mnist(self):
        """Download and validate MNIST dataset"""
        try:
            # Download training set
            train_dataset = datasets.MNIST(
                root=self.data_dir / 'mnist',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            
            # Download test set
            test_dataset = datasets.MNIST(
                root=self.data_dir / 'mnist',
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
            
            print(f"      ✅ MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return True
            
        except Exception as e:
            print(f"      ❌ Error downloading MNIST: {e}")
            return False
    
    def _download_cifar10(self):
        """Download and validate CIFAR-10 dataset"""
        try:
            # Download training set
            train_dataset = datasets.CIFAR10(
                root=self.data_dir / 'cifar10',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            
            # Download test set
            test_dataset = datasets.CIFAR10(
                root=self.data_dir / 'cifar10',
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
            
            print(f"      ✅ CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return True
            
        except Exception as e:
            print(f"      ❌ Error downloading CIFAR-10: {e}")
            return False
    
    def _download_femnist(self):
        """Download EMNIST as proxy for FEMNIST"""
        try:
            # Download EMNIST Letters split
            train_dataset = datasets.EMNIST(
                root=self.data_dir / 'femnist',
                split='letters',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            
            test_dataset = datasets.EMNIST(
                root=self.data_dir / 'femnist',
                split='letters',
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
            
            print(f"      ✅ FEMNIST (EMNIST): {len(train_dataset)} train, {len(test_dataset)} test samples")
            return True
            
        except Exception as e:
            print(f"      ❌ Error downloading FEMNIST: {e}")
            return False
    
    def validate_datasets(self):
        """Validate that all datasets are properly downloaded"""
        print("🔍 Validating datasets...")
        
        validation_results = {}
        
        # Validate MNIST
        validation_results['mnist'] = self._validate_mnist()
        
        # Validate CIFAR-10
        validation_results['cifar10'] = self._validate_cifar10()
        
        # Validate FEMNIST
        validation_results['femnist'] = self._validate_femnist()
        
        # Summary
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        print(f"\n📊 Validation Summary: {valid_count}/{total_count} datasets valid")
        
        return validation_results
    
    def _validate_mnist(self):
        """Validate MNIST dataset"""
        try:
            train_dataset = datasets.MNIST(
                root=self.data_dir / 'mnist',
                train=True,
                download=False,
                transform=transforms.ToTensor()
            )
            
            test_dataset = datasets.MNIST(
                root=self.data_dir / 'mnist',
                train=False,
                download=False,
                transform=transforms.ToTensor()
            )
            
            # Check data integrity
            sample_train = train_dataset[0]
            sample_test = test_dataset[0]
            
            assert sample_train[0].shape == (1, 28, 28), "Invalid MNIST train shape"
            assert sample_test[0].shape == (1, 28, 28), "Invalid MNIST test shape"
            assert 0 <= sample_train[1] <= 9, "Invalid MNIST train label"
            assert 0 <= sample_test[1] <= 9, "Invalid MNIST test label"
            
            print("   ✅ MNIST validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ MNIST validation failed: {e}")
            return False
    
    def _validate_cifar10(self):
        """Validate CIFAR-10 dataset"""
        try:
            train_dataset = datasets.CIFAR10(
                root=self.data_dir / 'cifar10',
                train=True,
                download=False,
                transform=transforms.ToTensor()
            )
            
            test_dataset = datasets.CIFAR10(
                root=self.data_dir / 'cifar10',
                train=False,
                download=False,
                transform=transforms.ToTensor()
            )
            
            # Check data integrity
            sample_train = train_dataset[0]
            sample_test = test_dataset[0]
            
            assert sample_train[0].shape == (3, 32, 32), "Invalid CIFAR-10 train shape"
            assert sample_test[0].shape == (3, 32, 32), "Invalid CIFAR-10 test shape"
            assert 0 <= sample_train[1] <= 9, "Invalid CIFAR-10 train label"
            assert 0 <= sample_test[1] <= 9, "Invalid CIFAR-10 test label"
            
            print("   ✅ CIFAR-10 validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ CIFAR-10 validation failed: {e}")
            return False
    
    def _validate_femnist(self):
        """Validate FEMNIST (EMNIST) dataset"""
        try:
            train_dataset = datasets.EMNIST(
                root=self.data_dir / 'femnist',
                split='letters',
                train=True,
                download=False,
                transform=transforms.ToTensor()
            )
            
            test_dataset = datasets.EMNIST(
                root=self.data_dir / 'femnist',
                split='letters',
                train=False,
                download=False,
                transform=transforms.ToTensor()
            )
            
            # Check data integrity
            sample_train = train_dataset[0]
            sample_test = test_dataset[0]
            
            assert sample_train[0].shape == (1, 28, 28), "Invalid FEMNIST train shape"
            assert sample_test[0].shape == (1, 28, 28), "Invalid FEMNIST test shape"
            assert 1 <= sample_train[1] <= 26, "Invalid FEMNIST train label"  # EMNIST letters: 1-26
            assert 1 <= sample_test[1] <= 26, "Invalid FEMNIST test label"
            
            print("   ✅ FEMNIST validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ FEMNIST validation failed: {e}")
            return False
    
    def analyze_dataset_properties(self, dataset_name):
        """Analyze properties of a specific dataset"""
        print(f"📊 Analyzing {dataset_name} dataset properties...")
        
        if dataset_name == 'mnist':
            return self._analyze_mnist()
        elif dataset_name == 'cifar10':
            return self._analyze_cifar10()
        elif dataset_name == 'femnist':
            return self._analyze_femnist()
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            return None
    
    def _analyze_mnist(self):
        """Analyze MNIST dataset properties"""
        train_dataset = datasets.MNIST(
            root=self.data_dir / 'mnist',
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )
        
        # Count class distribution
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(labels)
        
        analysis = {
            'dataset': 'mnist',
            'total_samples': len(train_dataset),
            'num_classes': 10,
            'input_shape': [1, 28, 28],
            'class_distribution': dict(class_counts),
            'class_balance': min(class_counts.values()) / max(class_counts.values())
        }
        
        print(f"   📈 Total samples: {analysis['total_samples']}")
        print(f"   📝 Classes: {analysis['num_classes']}")
        print(f"   🎯 Class balance: {analysis['class_balance']:.3f}")
        
        return analysis
    
    def _analyze_cifar10(self):
        """Analyze CIFAR-10 dataset properties"""
        train_dataset = datasets.CIFAR10(
            root=self.data_dir / 'cifar10',
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )
        
        # Count class distribution
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(labels)
        
        analysis = {
            'dataset': 'cifar10',
            'total_samples': len(train_dataset),
            'num_classes': 10,
            'input_shape': [3, 32, 32],
            'class_distribution': dict(class_counts),
            'class_balance': min(class_counts.values()) / max(class_counts.values())
        }
        
        print(f"   📈 Total samples: {analysis['total_samples']}")
        print(f"   📝 Classes: {analysis['num_classes']}")
        print(f"   🎯 Class balance: {analysis['class_balance']:.3f}")
        
        return analysis
    
    def _analyze_femnist(self):
        """Analyze FEMNIST dataset properties"""
        train_dataset = datasets.EMNIST(
            root=self.data_dir / 'femnist',
            split='letters',
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )
        
        # Count class distribution
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(labels)
        
        analysis = {
            'dataset': 'femnist',
            'total_samples': len(train_dataset),
            'num_classes': 26,
            'input_shape': [1, 28, 28],
            'class_distribution': dict(class_counts),
            'class_balance': min(class_counts.values()) / max(class_counts.values())
        }
        
        print(f"   📈 Total samples: {analysis['total_samples']}")
        print(f"   📝 Classes: {analysis['num_classes']}")
        print(f"   🎯 Class balance: {analysis['class_balance']:.3f}")
        
        return analysis
    
    def create_data_summary(self):
        """Create a comprehensive data summary"""
        print("📋 Creating data summary...")
        
        summary = {
            'datasets': {},
            'total_datasets': 0,
            'total_samples': 0,
            'preparation_timestamp': str(datetime.datetime.now())
        }
        
        for dataset_name in ['mnist', 'cifar10', 'femnist']:
            try:
                analysis = self.analyze_dataset_properties(dataset_name)
                if analysis:
                    summary['datasets'][dataset_name] = analysis
                    summary['total_samples'] += analysis['total_samples']
                    summary['total_datasets'] += 1
            except Exception as e:
                print(f"   ⚠️  Could not analyze {dataset_name}: {e}")
        
        # Save summary
        with open(self.data_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Summary saved: {summary['total_datasets']} datasets, {summary['total_samples']:,} samples")
        
        return summary
    
    def visualize_dataset_samples(self, dataset_name, num_samples=9):
        """Create visualization of dataset samples"""
        print(f"🎨 Creating sample visualization for {dataset_name}...")
        
        try:
            dataset = None
            if dataset_name == 'mnist':
                dataset = datasets.MNIST(
                    root=self.data_dir / 'mnist',
                    train=True,
                    download=False,
                    transform=transforms.ToTensor()
                )
            elif dataset_name == 'cifar10':
                dataset = datasets.CIFAR10(
                    root=self.data_dir / 'cifar10',
                    train=True,
                    download=False,
                    transform=transforms.ToTensor()
                )
            elif dataset_name == 'femnist':
                dataset = datasets.EMNIST(
                    root=self.data_dir / 'femnist',
                    split='letters',
                    train=True,
                    download=False,
                    transform=transforms.ToTensor()
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            if dataset is None:
                raise ValueError(f"Failed to load dataset: {dataset_name}")
            
            # Create visualization
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))
            fig.suptitle(f'{dataset_name.upper()} Dataset Samples', fontsize=16, fontweight='bold')
            
            for i in range(num_samples):
                row, col = i // 3, i % 3
                
                # Get random sample
                idx = np.random.randint(len(dataset))
                image, label = dataset[idx]
                
                # Convert tensor to numpy
                if image.shape[0] == 1:  # Grayscale
                    image = image.squeeze().numpy()
                    axes[row, col].imshow(image, cmap='gray')
                else:  # Color image
                    image = image.permute(1, 2, 0).numpy()
                    axes[row, col].imshow(image)
                
                axes[row, col].set_title(f'Label: {label}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_dir = self.data_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            plt.savefig(viz_dir / f'{dataset_name}_samples.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Visualization saved for {dataset_name}")
            
        except Exception as e:
            print(f"   ❌ Could not create visualization for {dataset_name}: {e}")
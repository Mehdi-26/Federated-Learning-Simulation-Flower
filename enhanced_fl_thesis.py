beta1 = self.config.get('algorithm_params', {}).get('fedadam', {}).get('beta1', 0.9)
        beta2 = self.config.get('algorithm_params', {}).get('fedadam', {}).get('beta2', 0.999)
        tau = self.fedadam_tau
        
        if self.fedadam_m is not None and self.fedadam_v is not None:
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if name in self.fedadam_m and name in self.fedadam_v:
                        # Compute pseudo-gradient (difference from previous round)
                        if round_num > 0:
                            # Update momentum and variance
                            self.fedadam_m[name] = beta1 * self.fedadam_m[name] + (1 - beta1) * param.grad if param.grad is not None else self.fedadam_m[name]
                            self.fedadam_v[name] = beta2 * self.fedadam_v[name] + (1 - beta2) * (param.grad ** 2) if param.grad is not None else self.fedadam_v[name]
                            
                            # Bias correction
                            m_hat = self.fedadam_m[name] / (1 - beta1 ** (round_num + 1))
                            v_hat = self.fedadam_v[name] / (1 - beta2 ** (round_num + 1))
                            
                            # Apply adaptive update
                            param.data -= tau * m_hat / (torch.sqrt(v_hat) + 1e-8)
        
        return global_model

    def _scaffold_aggregation(self, global_model, client_weights, client_samples):
        """SCAFFOLD server aggregation with control variates"""
        # Perform standard FedAvg aggregation
        global_model = self._fedavg_aggregation(global_model, client_weights, client_samples)
        
        # Update global control variate
        if self.scaffold_global_c is not None and self.scaffold_client_c is not None:
            num_clients = len(self.scaffold_client_c)
            for name in self.scaffold_global_c.keys():
                self.scaffold_global_c[name] = sum(client_c[name] for client_c in self.scaffold_client_c) / num_clients
        
        return global_model

    def _coop_aggregation(self, global_model, client_weights, client_samples):
        """COOP (Cooperative) aggregation with enhanced collaboration"""
        # Enhanced weighted averaging with cooperation factors
        total_samples = sum(client_samples)
        alpha = self.config.get('algorithm_params', {}).get('coop', {}).get('alpha', 0.1)
        
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Standard weighted aggregation
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight_factor = num_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight_factor * client_weight[key]
        
        # Apply cooperation enhancement
        current_params = {name: param.clone() for name, param in global_model.named_parameters()}
        for key in aggregated_weights.keys():
            if key in current_params:
                aggregated_weights[key] = (1 - alpha) * aggregated_weights[key] + alpha * current_params[key]
        
        global_model.load_state_dict(aggregated_weights)
        return global_model

    def analyze_gradient_diversity(self, client_gradients):
        """Analyze gradient diversity across clients"""
        if len(client_gradients) < 2:
            return 0.0, 0.0, 0.0
        
        # Flatten all gradients
        flattened_gradients = []
        for grad_dict in client_gradients:
            flattened = []
            for param_name, grad in grad_dict.items():
                if grad is not None:
                    flattened.extend(grad.flatten().cpu().numpy())
            if flattened:
                flattened_gradients.append(np.array(flattened))
        
        if not flattened_gradients:
            return 0.0, 0.0, 0.0
        
        # Calculate cosine similarity between gradients
        similarities = []
        for i in range(len(flattened_gradients)):
            for j in range(i + 1, len(flattened_gradients)):
                norm_i = np.linalg.norm(flattened_gradients[i])
                norm_j = np.linalg.norm(flattened_gradients[j])
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(flattened_gradients[i], flattened_gradients[j]) / (norm_i * norm_j)
                    similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        std_similarity = np.std(similarities) if similarities else 0.0
        
        # Calculate gradient magnitude variance
        magnitudes = [np.linalg.norm(grad) for grad in flattened_gradients]
        magnitude_variance = np.var(magnitudes) if magnitudes else 0.0
        
        return avg_similarity, std_similarity, magnitude_variance

    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total_loss = 0.0
        total_samples = 0
        criterion = nn.NLLLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
        
        accuracy = correct / total_samples
        avg_loss = total_loss / total_samples
        return accuracy, avg_loss

    def _calculate_model_drift(self, client_weights: List, global_weights: Dict) -> float:
        """Calculate model drift between clients and global model"""
        if not client_weights:
            return 0.0
        
        total_drift = 0.0
        num_params = 0
        
        for client_weight in client_weights:
            for key in client_weight.keys():
                if key in global_weights:
                    drift = torch.norm(client_weight[key] - global_weights[key]).item()
                    total_drift += drift
                    num_params += 1
        
        return total_drift / num_params if num_params > 0 else 0.0

    def _calculate_fairness(self, client_accuracies: List[float]) -> float:
        """Calculate fairness score using Gini coefficient"""
        if len(client_accuracies) <= 1:
            return 1.0
        
        sorted_acc = sorted(client_accuracies)
        n = len(sorted_acc)
        cumsum = np.cumsum(sorted_acc)
        
        if cumsum[-1] == 0:
            return 1.0
        
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return 1.0 - gini

    def _calculate_model_stability(self, client_weights: List) -> float:
        """Calculate model stability across clients"""
        if len(client_weights) < 2:
            return 1.0
        
        stability_scores = []
        for i in range(len(client_weights)):
            for j in range(i + 1, len(client_weights)):
                param_diff = 0.0
                param_count = 0
                
                for key in client_weights[i].keys():
                    if key in client_weights[j]:
                        diff = torch.norm(client_weights[i][key] - client_weights[j][key]).item()
                        param_diff += diff
                        param_count += 1
                
                if param_count > 0:
                    stability_scores.append(param_diff / param_count)
        
        avg_diff = np.mean(stability_scores) if stability_scores else 0.0
        return 1.0 / (1.0 + avg_diff)

    def _calculate_consensus_score(self, client_accuracies: List[float], global_accuracy: float) -> float:
        """Calculate consensus score between clients and global model"""
        if not client_accuracies:
            return 0.0
        
        client_mean = np.mean(client_accuracies)
        client_std = np.std(client_accuracies)
        
        consensus = 1.0 / (1.0 + client_std)
        alignment = 1.0 - abs(client_mean - global_accuracy)
        
        return consensus * alignment

    def _calculate_convergence_rate(self, accuracy_history: List[float], current_acc: float) -> float:
        """Calculate convergence rate"""
        if len(accuracy_history) < 1:
            return 0.0
        return current_acc - accuracy_history[-1] if accuracy_history else 0.0

    def _calculate_privacy_budget(self, round_num: int, epsilon: float) -> float:
        """Calculate privacy budget consumed"""
        return epsilon * (round_num + 1)

    def _calculate_final_metrics(self, metrics: Dict) -> Dict:
        """Calculate comprehensive final metrics for academic analysis"""
        final_metrics = {}
        
        # Basic final metrics
        final_metrics['final_accuracy'] = metrics['test_accuracy'][-1] if metrics['test_accuracy'] else 0.0
        final_metrics['final_loss'] = metrics['test_loss'][-1] if metrics['test_loss'] else float('inf')
        final_metrics['total_communication_cost'] = metrics['communication_cost'][-1] if metrics['communication_cost'] else 0
        final_metrics['convergence_round'] = len(metrics['rounds'])
        final_metrics['total_time'] = sum(metrics['round_times'])
        
        # Advanced metrics
        final_metrics['avg_model_drift'] = np.mean(metrics['model_drift']) if metrics['model_drift'] else 0.0
        final_metrics['avg_fairness'] = np.mean(metrics['client_fairness']) if metrics['client_fairness'] else 0.0
        final_metrics['avg_gradient_diversity'] = np.mean(metrics['gradient_diversity']) if metrics['gradient_diversity'] else 0.0
        final_metrics['avg_participation_rate'] = np.mean(metrics['client_participation']) if metrics['client_participation'] else 0.0
        final_metrics['avg_model_stability'] = np.mean(metrics['model_stability']) if metrics['model_stability'] else 0.0
        final_metrics['avg_consensus_score'] = np.mean(metrics['consensus_metrics']) if metrics['consensus_metrics'] else 0.0
        final_metrics['accuracy_variance'] = np.var(metrics['test_accuracy']) if metrics['test_accuracy'] else 0.0
        final_metrics['client_accuracy_variance'] = np.mean(metrics['client_accuracy_variance']) if metrics['client_accuracy_variance'] else 0.0
        
        # Convergence analysis
        if len(metrics['test_accuracy']) > 1:
            accuracy_diffs = np.diff(metrics['test_accuracy'])
            final_metrics['avg_convergence_rate'] = np.mean(accuracy_diffs)
            final_metrics['convergence_stability'] = 1.0 / (1.0 + np.std(accuracy_diffs))
        else:
            final_metrics['avg_convergence_rate'] = 0.0
            final_metrics['convergence_stability'] = 0.0
        
        return final_metrics

    def run_comprehensive_experiment(self):
        """Run comprehensive experiments with academic analysis"""
        logger.info("Starting Enhanced Comprehensive FL Experiment with Flower Algorithms")
        
        all_results = {}
        experiment_id = f"enhanced_thesis_experiment_{int(time.time())}"
        
        # Run centralized baselines
        centralized_results = {}
        for dataset in self.config['datasets']:
            logger.info(f"Running centralized baseline for {dataset}")
            centralized_results[dataset] = self.run_centralized_baseline(dataset)
        
        # Get distributions from config
        distributions = self.config.get('data_distributions', ['iid', 'non_iid'])
        
        # Calculate total experiments
        total_experiments = 0
        for algorithm in self.config['algorithms']:
            for dataset in self.config['datasets']:
                for distribution in distributions:
                    beta_values = [1.0] if distribution == 'iid' else self.config.get('beta_values', [0.1, 0.5, 1.0])
                    total_experiments += len(beta_values)
        
        logger.info(f"Total federated experiments: {total_experiments}")
        logger.info(f"Total centralized baselines: {len(self.config['datasets'])}")
        
        experiment_count = 0
        
        # Run federated experiments
        for algorithm in self.config['algorithms']:
            all_results[algorithm] = {}
            
            for dataset in self.config['datasets']:
                all_results[algorithm][dataset] = {}
                
                for distribution in distributions:
                    all_results[algorithm][dataset][distribution] = {}
                    
                    beta_values = [1.0] if distribution == 'iid' else self.config.get('beta_values', [0.1, 0.5, 1.0])
                    
                    for beta in beta_values:
                        experiment_count += 1
                        logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                                  f"{algorithm} on {dataset} ({distribution}, beta={beta})")
                        
                        # Run enhanced experiment
                        result = self.run_enhanced_algorithm(algorithm, dataset, distribution, beta)
                        all_results[algorithm][dataset][distribution][f'beta_{beta}'] = result
        
        # Generate comprehensive academic analysis
        academic_analysis = self._generate_academic_analysis(all_results, centralized_results)
        
        # Generate research question analysis
        research_analysis = self._analyze_research_questions(all_results, centralized_results)
        
        # Save comprehensive results
        self._save_enhanced_results(all_results, academic_analysis, research_analysis, 
                                  centralized_results, experiment_id)
        
        # Generate enhanced academic visualizations
        self._generate_academic_dashboards(all_results, academic_analysis, research_analysis, 
                                         centralized_results, experiment_id)
        
        return all_results, academic_analysis, research_analysis

    def _generate_academic_analysis(self, fl_results: Dict, centralized_results: Dict) -> Dict:
        """Generate comprehensive academic analysis"""
        analysis = {
            'statistical_significance': {},
            'algorithm_rankings': {},
            'heterogeneity_impact': {},
            'robustness_analysis': {},
            'efficiency_analysis': {},
            'scalability_insights': {},
            'convergence_analysis': {},
            'fairness_analysis': {}
        }
        
        # Statistical significance testing
        for dataset in self.config['datasets']:
            analysis['statistical_significance'][dataset] = {}
            
            # Collect accuracy data for statistical tests
            algorithm_accuracies = {}
            for algorithm in self.config['algorithms']:
                accuracies = []
                if algorithm in fl_results:
                    for dist in fl_results[algorithm].get(dataset, {}):
                        for beta_key, result in fl_results[algorithm][dataset][dist].items():
                            if 'test_accuracy' in result:
                                accuracies.extend(result['test_accuracy'])
                algorithm_accuracies[algorithm] = accuracies
            
            # Perform pairwise statistical tests
            algorithms = list(algorithm_accuracies.keys())
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    alg1, alg2 = algorithms[i], algorithms[j]
                    if algorithm_accuracies[alg1] and algorithm_accuracies[alg2]:
                        try:
                            statistic, p_value = stats.ttest_ind(
                                algorithm_accuracies[alg1], 
                                algorithm_accuracies[alg2]
                            )
                            analysis['statistical_significance'][dataset][f'{alg1}_vs_{alg2}'] = {
                                'statistic': float(statistic),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {alg1} vs {alg2}: {e}")
        
        # Algorithm ranking analysis
        algorithm_scores = {}
        for algorithm in self.config['algorithms']:
            scores = []
            if algorithm in fl_results:
                for dataset in self.config['datasets']:
                    if dataset in fl_results[algorithm]:
                        for dist in fl_results[algorithm][dataset]:
                            for beta_key, result in fl_results[algorithm][dataset][dist].items():
                                accuracy_score = result.get('final_accuracy', 0.0)
                                efficiency_score = 1.0 - min(result.get('total_communication_cost', 0) / 1e9, 1.0)
                                fairness_score = result.get('avg_fairness', 0.0)
                                stability_score = result.get('avg_model_stability', 0.0)
                                
                                composite_score = (accuracy_score * 0.4 + efficiency_score * 0.2 + 
                                                 fairness_score * 0.2 + stability_score * 0.2)
                                scores.append(composite_score)
            
            algorithm_scores[algorithm] = {
                'mean_score': np.mean(scores) if scores else 0.0,
                'std_score': np.std(scores) if scores else 0.0,
                'scores': scores
            }
        
        analysis['algorithm_rankings'] = dict(sorted(
            algorithm_scores.items(), 
            key=lambda x: x[1]['mean_score'], 
            reverse=True
        ))
        
        return analysis

    def _analyze_research_questions(self, fl_results: Dict, centralized_results: Dict) -> Dict:
        """Analyze specific research questions"""
        research_analysis = {
            'fl_vs_centralized': {},
            'non_iid_impact': {},
            'device_reliability': {},
            'security_vs_accuracy_tradeoff': {},
            'scalability_findings': {}
        }
        
        # Research Question 1: FL vs Centralized Learning
        for dataset in self.config['datasets']:
            if dataset in centralized_results:
                centralized_acc = centralized_results[dataset]['final_accuracy']
                centralized_time = centralized_results[dataset]['total_time']
                
                fl_accuracies = []
                fl_times = []
                fl_comm_costs = []
                
                for algorithm in self.config['algorithms']:
                    if algorithm in fl_results and dataset in fl_results[algorithm]:
                        for dist in fl_results[algorithm][dataset]:
                            for beta_key, result in fl_results[algorithm][dataset][dist].items():
                                fl_accuracies.append(result.get('final_accuracy', 0.0))
                                fl_times.append(result.get('total_time', 0.0))
                                fl_comm_costs.append(result.get('total_communication_cost', 0))
                
                research_analysis['fl_vs_centralized'][dataset] = {
                    'centralized_accuracy': centralized_acc,
                    'fl_avg_accuracy': np.mean(fl_accuracies) if fl_accuracies else 0.0,
                    'fl_accuracy_std': np.std(fl_accuracies) if fl_accuracies else 0.0,
                    'accuracy_gap': centralized_acc - np.mean(fl_accuracies) if fl_accuracies else 0.0,
                    'centralized_time': centralized_time,
                    'fl_avg_time': np.mean(fl_times) if fl_times else 0.0,
                    'time_efficiency': centralized_time / np.mean(fl_times) if fl_times and np.mean(fl_times) > 0 else 1.0,
                    'communication_overhead': np.mean(fl_comm_costs) if fl_comm_costs else 0.0,
                    'privacy_preserved': True,
                    'data_centralization_required': False
                }
        
        # Research Question 2: Non-IID Impact Analysis
        for dataset in self.config['datasets']:
            iid_results = []
            non_iid_results = {}
            
            for algorithm in self.config['algorithms']:
                if algorithm in fl_results and dataset in fl_results[algorithm]:
                    if 'iid' in fl_results[algorithm][dataset]:
                        for beta_key, result in fl_results[algorithm][dataset]['iid'].items():
                            iid_results.append(result.get('final_accuracy', 0.0))
                    
                    if 'non_iid' in fl_results[algorithm][dataset]:
                        for beta_key, result in fl_results[algorithm][dataset]['non_iid'].items():
                            beta = float(beta_key.split('_')[1])
                            if beta not in non_iid_results:
                                non_iid_results[beta] = []
                            non_iid_results[beta].append(result.get('final_accuracy', 0.0))
            
            research_analysis['non_iid_impact'][dataset] = {
                'iid_avg_accuracy': np.mean(iid_results) if iid_results else 0.0,
                'non_iid_results': {
                    beta: {
                        'avg_accuracy': np.mean(accs),
                        'accuracy_degradation': np.mean(iid_results) - np.mean(accs) if iid_results else 0.0,
                        'relative_degradation': (np.mean(iid_results) - np.mean(accs)) / np.mean(iid_results) * 100 if iid_results and np.mean(iid_results) > 0 else 0.0
                    }
                    for beta, accs in non_iid_results.items()
                }
            }
        
        # Research Question 3: Device Reliability Impact
        dropout_analysis = {}
        for algorithm in self.config['algorithms']:
            if algorithm in fl_results:
                for dataset in self.config['datasets']:
                    if dataset in fl_results[algorithm]:
                        for dist in fl_results[algorithm][dataset]:
                            for beta_key, result in fl_results[algorithm][dataset][dist].items():
                                if 'dropout_analysis' in result and result['dropout_analysis']:
                                    key = f"{algorithm}_{dataset}_{dist}_{beta_key}"
                                    dropout_analysis[key] = {
                                        'baseline_accuracy': result['test_accuracy'][0] if result.get('test_accuracy') else 0.0,
                                        'final_accuracy': result.get('final_accuracy', 0.0),
                                        'dropout_rounds': list(result['dropout_analysis'].keys()),
                                        'dropout_impact': {}
                                    }
                                    
                                    for round_num, dropout_info in result['dropout_analysis'].items():
                                        test_accuracy = result.get('test_accuracy', [])
                                        if round_num < len(test_accuracy):
                                            before_acc = test_accuracy[round_num - 1] if round_num > 0 else 0.0
                                            after_acc = test_accuracy[round_num] if round_num < len(test_accuracy) else 0.0
                                            dropout_analysis[key]['dropout_impact'][round_num] = {
                                                'dropout_rate': dropout_info['dropout_rate'],
                                                'accuracy_before': before_acc,
                                                'accuracy_after': after_acc,
                                                'accuracy_drop': before_acc - after_acc
                                            }
        
        research_analysis['device_reliability'] = dropout_analysis
        
        return research_analysis

    def _save_enhanced_results(self, fl_results: Dict, academic_analysis: Dict, 
                             research_analysis: Dict, centralized_results: Dict, experiment_id: str):
        """Save enhanced results with academic focus"""
        output_dir = Path(f"results/enhanced_thesis_{experiment_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all results
        with open(output_dir / 'federated_results.json', 'w') as f:
            json.dump(fl_results, f, indent=2, default=str)
        
        with open(output_dir / 'centralized_results.json', 'w') as f:
            json.dump(centralized_results, f, indent=2, default=str)
        
        with open(output_dir / 'academic_analysis.json', 'w') as f:
            json.dump(academic_analysis, f, indent=2, default=str)
        
        with open(output_dir / 'research_analysis.json', 'w') as f:
            json.dump(research_analysis, f, indent=2, default=str)
        
        # Create comprehensive CSV for statistical analysis
        self._create_enhanced_csv(fl_results, centralized_results, output_dir)
        
        # Save configuration
        with open(output_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Enhanced results saved to {output_dir}")

    def _create_enhanced_csv(self, fl_results: Dict, centralized_results: Dict, output_dir: Path):
        """Create enhanced CSV with all metrics for statistical analysis"""
        # Federated learning results
        fl_rows = []
        for algorithm in fl_results:
            for dataset in fl_results[algorithm]:
                for distribution in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][distribution].items():
                        beta = float(beta_key.split('_')[1])
                        
                        base_row = {
                            'experiment_type': 'federated',
                            'algorithm': algorithm,
                            'dataset': dataset,
                            'distribution': distribution,
                            'beta': beta,
                            'final_accuracy': result.get('final_accuracy', 0.0),
                            'final_loss': result.get('final_loss', 0.0),
                            'convergence_round': result.get('convergence_round', 0),
                            'total_communication_cost': result.get('total_communication_cost', 0),
                            'total_time': result.get('total_time', 0.0),
                            'avg_model_drift': result.get('avg_model_drift', 0.0),
                            'avg_fairness': result.get('avg_fairness', 0.0),
                            'avg_gradient_diversity': result.get('avg_gradient_diversity', 0.0),
                            'avg_participation_rate': result.get('avg_participation_rate', 0.0),
                            'avg_model_stability': result.get('avg_model_stability', 0.0),
                            'avg_consensus_score': result.get('avg_consensus_score', 0.0),
                            'accuracy_variance': result.get('accuracy_variance', 0.0),
                            'client_accuracy_variance': result.get('client_accuracy_variance', 0.0),
                            'avg_convergence_rate': result.get('avg_convergence_rate', 0.0),
                            'convergence_stability': result.get('convergence_stability', 0.0)
                        }
                        
                        # Add heterogeneity metrics
                        if 'heterogeneity_metrics' in result:
                            base_row.update({
                                f'heterogeneity_{k}': v for k, v in result['heterogeneity_metrics'].items()
                            })
                        
                        fl_rows.append(base_row)
        
        # Centralized learning results
        centralized_rows = []
        for dataset, result in centralized_results.items():
            centralized_rows.append({
                'experiment_type': 'centralized',
                'algorithm': 'centralized_sgd',
                'dataset': dataset,
                'distribution': 'iid',
                'beta': 1.0,
                'final_accuracy': result.get('final_accuracy', 0.0),
                'final_loss': result.get('final_loss', 0.0),
                'convergence_round': result.get('convergence_round', 0),
                'total_communication_cost': 0,
                'total_time': result.get('total_time', 0.0),
                'avg_model_drift': 0.0,
                'avg_fairness': 1.0,
                'avg_gradient_diversity': 0.0,
                'avg_participation_rate': 1.0,
                'avg_model_stability': 1.0,
                'avg_consensus_score': 1.0,
                'accuracy_variance': np.var(result.get('test_accuracy', [])) if result.get('test_accuracy') else 0.0,
                'client_accuracy_variance': 0.0,
                'avg_convergence_rate': np.mean(np.diff(result['test_accuracy'])) if len(result.get('test_accuracy', [])) > 1 else 0.0,
                'convergence_stability': 1.0 / (1.0 + np.std(np.diff(result['test_accuracy']))) if len(result.get('test_accuracy', [])) > 1 else 0.0
            })
        
        # Combine and save
        all_rows = fl_rows + centralized_rows
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / 'comprehensive_results.csv', index=False)
        
        # Save separate files for specific analyses
        pd.DataFrame(fl_rows).to_csv(output_dir / 'federated_only_results.csv', index=False)
        pd.DataFrame(centralized_rows).to_csv(output_dir / 'centralized_only_results.csv', index=False)

    def _generate_academic_dashboards(self, fl_results: Dict, academic_analysis: Dict, 
                                    research_analysis: Dict, centralized_results: Dict, experiment_id: str):
        """Generate comprehensive academic dashboards like the examples shown"""
        output_dir = Path(f"results/enhanced_thesis_{experiment_id}")
        plots_dir = output_dir / "academic_dashboards"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating comprehensive academic dashboards...")
        
        try:
            # 1. Create Training Progress Dashboard (like Image 5)
            self._create_training_progress_dashboard(fl_results, plots_dir)
            
            # 2. Create Algorithm Summary Dashboard (like Image 4)
            self._create_algorithm_summary_dashboard(fl_results, plots_dir)
            
            # 3. Create Performance Metrics Dashboard (like Image 1)
            self._create_performance_metrics_dashboard(fl_results, plots_dir)
            
            # 4. Create Communication and Fairness Dashboard (like Images 2&3)
            self._create_communication_fairness_dashboard(fl_results, plots_dir)
            
            # 5. Create Research Questions Dashboard
            self._create_research_questions_dashboard(fl_results, research_analysis, centralized_results, plots_dir)
            
            logger.info("Academic dashboards generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate academic dashboards: {e}")
        
        return plots_dir

    def _create_training_progress_dashboard(self, fl_results: Dict, output_dir: Path):
        """Create training progress dashboard similar to Image 5"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Color scheme for algorithms
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        algorithm_colors = {}
        
        for i, algorithm in enumerate(fl_results.keys()):
            algorithm_colors[algorithm] = colors[i % len(colors)]
        
        # Plot 1: Test Accuracy Over Rounds
        ax1 = axes[0, 0]
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'test_accuracy' in result and result['test_accuracy']:
                            rounds = result['rounds']
                            accuracy = result['test_accuracy']
                            ax1.plot(rounds, accuracy, label=algorithm, 
                                   color=algorithm_colors[algorithm], linewidth=2)
        
        ax1.set_title('Test Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Loss Over Rounds
        ax2 = axes[0, 1]
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'train_loss' in result and result['train_loss']:
                            rounds = result['rounds']
                            loss = result['train_loss']
                            ax2.plot(rounds, loss, label=algorithm, 
                                   color=algorithm_colors[algorithm], linewidth=2)
        
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Communication
        ax3 = axes[1, 0]
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'communication_cost' in result and result['communication_cost']:
                            rounds = result['rounds']
                            comm_cost = [cost / 1000 for cost in result['communication_cost']]  # Convert to K bytes
                            ax3.plot(rounds, comm_cost, label=algorithm, 
                                   color=algorithm_colors[algorithm], linewidth=2)
        
        ax3.set_title('Cumulative Communication', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Bytes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence Measure
        ax4 = axes[1, 1]
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'test_accuracy' in result and result['test_accuracy']:
                            rounds = result['rounds']
                            # Calculate convergence measure (1 - accuracy gap from max)
                            max_acc = max(result['test_accuracy'])
                            convergence = [1 - (max_acc - acc) for acc in result['test_accuracy']]
                            ax4.plot(rounds, convergence, label=algorithm, 
                                   color=algorithm_colors[algorithm], linewidth=2)
        
        ax4.set_title('Convergence Measure', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_algorithm_summary_dashboard(self, fl_results: Dict, output_dir: Path):
        """Create algorithm summary dashboard similar to Image 4"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.3)
        
        # Colors for algorithms
        colors = ['#87CEEB', '#F08080', '#90EE90', '#FFB347']
        
        # Collect data for all algorithms
        algorithm_data = {}
        for algorithm in fl_results.keys():
            algorithm_data[algorithm] = {
                'final_accuracy': [],
                'comm_cost': [],
                'convergence_round': [],
                'training_time': []
            }
            
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        algorithm_data[algorithm]['final_accuracy'].append(result.get('final_accuracy', 0.0))
                        algorithm_data[algorithm]['comm_cost'].append(result.get('total_communication_cost', 0))
                        algorithm_data[algorithm]['convergence_round'].append(result.get('convergence_round', 0))
                        algorithm_data[algorithm]['training_time'].append(result.get('total_time', 0.0))
        
        # Plot 1: Final Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        algorithms = list(algorithm_data.keys())
        accuracies = [np.mean(algorithm_data[alg]['final_accuracy']) for alg in algorithms]
        
        bars = ax1.bar(algorithms, accuracies, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.0)
        
        # Plot 2: Communication Cost
        ax2 = fig.add_subplot(gs[0, 1])
        comm_costs = [np.mean(algorithm_data[alg]['comm_cost']) / 1e6 for alg in algorithms]  # Convert to MB
        
        bars = ax2.bar(algorithms, comm_costs, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
        ax2.set_title('Communication Cost', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Bytes (MB)')
        
        # Plot 3: Training Progress (Multi-line)
        ax3 = fig.add_subplot(gs[1, :])
        for i, algorithm in enumerate(algorithms):
            # Get a representative training curve
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'test_accuracy' in result and result['test_accuracy']:
                            rounds = result['rounds']
                            accuracy = result['test_accuracy']
                            ax3.plot(rounds, accuracy, label=algorithm, 
                                   color=colors[i], linewidth=2, marker='o', markersize=4)
                            break
                    break
                break
        
        ax3.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Accuracy')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for algorithm in algorithms:
            final_acc = np.mean(algorithm_data[algorithm]['final_accuracy'])
            comm_cost = int(np.mean(algorithm_data[algorithm]['comm_cost']))
            conv_round = int(np.mean(algorithm_data[algorithm]['convergence_round']))
            train_time = int(np.mean(algorithm_data[algorithm]['training_time']))
            
            table_data.append([algorithm, f'{final_acc:.3f}', f'{comm_cost:,}', conv_round, f'{train_time}s'])
        
        headers = ['Algorithm', 'Final Accuracy', 'Comm Cost (Bytes)', 'Convergence Round', 'Training Time']
        table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Federated Learning Experiment Summary', fontsize=16, fontweight='bold')
        plt.savefig(output_dir / 'algorithm_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_metrics_dashboard(self, fl_results: Dict, output_dir: Path):
        """Create performance metrics dashboard similar to Image 1"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colors for different metrics
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Collect all data for plotting
        all_data = []
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        all_data.append({
                            'algorithm': algorithm,
                            'dataset': dataset,
                            'distribution': dist,
                            'beta': beta_key,
                            'final_accuracy': result.get('final_accuracy', 0.0),
                            'convergence_round': result.get('convergence_round', 0),
                            'communication_cost': result.get('total_communication_cost', 0),
                            'training_time': result.get('total_time', 0.0)
                        })
        
        df = pd.DataFrame(all_data)
        
        # Plot 1: Final Accuracy by Algorithm
        ax1 = axes[0, 0]
        experiment_labels = [f"{row['algorithm']}\n{row['dataset'].upper()}" for _, row in df.iterrows()]
        
        bars = ax1.bar(range(len(df)), df['final_accuracy'], color=colors[:len(df)], alpha=0.8)
        ax1.set_title('Final Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(experiment_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Convergence Round
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(df)), df['convergence_round'], color=colors[:len(df)], alpha=0.8)
        ax2.set_title('Convergence Round', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Rounds')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(experiment_labels, rotation=45, ha='right', fontsize=8)
        
        # Plot 3: Communication Cost
        ax3 = axes[1, 0]
        comm_costs_mb = df['communication_cost'] / 1e6  # Convert to MB
        bars = ax3.bar(range(len(df)), comm_costs_mb, color=colors[:len(df)], alpha=0.8)
        ax3.set_title('Communication Cost', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Cost (MB)')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(experiment_labels, rotation=45, ha='right', fontsize=8)
        
        # Plot 4: Training Time
        ax4 = axes[1, 1]
        bars = ax4.bar(range(len(df)), df['training_time'], color=colors[:len(df)], alpha=0.8)
        ax4.set_title('Training Time', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(experiment_labels, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_metrics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_communication_fairness_dashboard(self, fl_results: Dict, output_dir: Path):
        """Create communication and fairness dashboard similar to Images 2&3"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Communication Efficiency Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        algorithm_colors = {}
        
        for i, algorithm in enumerate(fl_results.keys()):
            algorithm_colors[algorithm] = colors[i % len(colors)]
            
            comm_costs = []
            accuracies = []
            
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        comm_costs.append(result.get('total_communication_cost', 0) / 1e6)  # MB
                        accuracies.append(result.get('final_accuracy', 0.0))
            
            if comm_costs and accuracies:
                ax1.scatter(comm_costs, accuracies, label=algorithm, 
                           color=algorithm_colors[algorithm], s=80, alpha=0.7, edgecolors='black')
        
        ax1.set_title('Communication Efficiency', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Communication Cost (Bytes)')
        ax1.set_ylabel('Final Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Communication Cost by Data Heterogeneity (Box Plot)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Collect data for box plot
        beta_data = {}
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    if dist == 'non_iid':  # Only non-IID for heterogeneity analysis
                        for beta_key, result in fl_results[algorithm][dataset][dist].items():
                            beta = beta_key.split('_')[1]
                            if beta not in beta_data:
                                beta_data[beta] = []
                            beta_data[beta].append(result.get('total_communication_cost', 0))
        
        if beta_data:
            betas = sorted(beta_data.keys(), key=float)
            data_for_box = [beta_data[beta] for beta in betas]
            
            bp = ax2.boxplot(data_for_box, labels=[f'β={beta}' for beta in betas], patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_title('Communication Cost by Data Heterogeneity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dirichlet Parameter (β)')
        ax2.set_ylabel('Communication Cost (Bytes)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fairness Over Time
        ax3 = fig.add_subplot(gs[1, :])
        
        for algorithm in fl_results.keys():
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        if 'client_fairness' in result and result['client_fairness']:
                            rounds = result['rounds']
                            fairness = result['client_fairness']
                            ax3.plot(rounds, fairness, label=f'{algorithm}', 
                                   color=algorithm_colors.get(algorithm, colors[0]), 
                                   linewidth=2, alpha=0.8)
                            break
                    break
                break
        
        ax3.set_title('Fairness Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Gini Coefficient')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Communication Efficiency and Fairness Analysis', fontsize=16, fontweight='bold')
        plt.savefig(output_dir / 'communication_fairness_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_research_questions_dashboard(self, fl_results: Dict, research_analysis: Dict, 
                                           centralized_results: Dict, output_dir: Path):
        """Create research questions dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: FL vs Centralized Comparison
        ax1 = axes[0, 0]
        if research_analysis.get('fl_vs_centralized'):
            datasets = list(research_analysis['fl_vs_centralized'].keys())
            centralized_accs = [research_analysis['fl_vs_centralized'][d]['centralized_accuracy'] for d in datasets]
            fl_accs = [research_analysis['fl_vs_centralized'][d]['fl_avg_accuracy'] for d in datasets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            ax1.bar(x - width/2, centralized_accs, width, label='Centralized', alpha=0.8, color='#4CAF50')
            ax1.bar(x + width/2, fl_accs, width, label='Federated', alpha=0.8, color='#2196F3')
            
            # Add value labels
            for i, (cent, fl) in enumerate(zip(centralized_accs, fl_accs)):
                ax1.text(i - width/2, cent + 0.01, f'{cent:.3f}', ha='center', va='bottom', fontweight='bold')
                ax1.text(i + width/2, fl + 0.01, f'{fl:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_title('FL vs Centralized Learning', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_xticks(x)
            ax1.set_xticklabels([d.upper() for d in datasets])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Non-IID Impact
        ax2 = axes[0, 1]
        if research_analysis.get('non_iid_impact'):
            for dataset, data in research_analysis['non_iid_impact'].items():
                if 'non_iid_results' in data:
                    betas = list(data['non_iid_results'].keys())
                    degradations = [data['non_iid_results'][beta]['relative_degradation'] for beta in betas]
                    ax2.plot(betas, degradations, marker='o', linewidth=3, markersize=8, 
                           label=dataset.upper())
        
        ax2.set_title('Non-IID Impact on Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Beta (Heterogeneity Level)')
        ax2.set_ylabel('Performance Degradation (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Device Dropout Impact
        ax3 = axes[1, 0]
        dropout_rates = []
        accuracy_drops = []
        
        if research_analysis.get('device_reliability'):
            for key, data in research_analysis['device_reliability'].items():
                if 'dropout_impact' in data:
                    for round_num, impact in data['dropout_impact'].items():
                        dropout_rates.append(impact['dropout_rate'])
                        accuracy_drops.append(impact['accuracy_drop'])
        
        if dropout_rates and accuracy_drops:
            ax3.scatter(dropout_rates, accuracy_drops, alpha=0.7, s=80, edgecolors='black', color='#FF5722')
            
            # Add trend line
            if len(dropout_rates) > 1:
                z = np.polyfit(dropout_rates, accuracy_drops, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(dropout_rates), p(sorted(dropout_rates)), 
                       color='red', linewidth=2, linestyle='--', label='Trend')
                ax3.legend()
        
        ax3.set_title('Device Dropout Impact', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Dropout Rate')
        ax3.set_ylabel('Accuracy Drop')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Algorithm Rankings
        ax4 = axes[1, 1]
        algorithms = list(fl_results.keys())
        final_accuracies = []
        
        for algorithm in algorithms:
            accuracies = []
            for dataset in fl_results[algorithm]:
                for dist in fl_results[algorithm][dataset]:
                    for beta_key, result in fl_results[algorithm][dataset][dist].items():
                        accuracies.append(result.get('final_accuracy', 0.0))
            final_accuracies.append(np.mean(accuracies) if accuracies else 0.0)
        
        # Sort by accuracy
        sorted_data = sorted(zip(algorithms, final_accuracies), key=lambda x: x[1], reverse=True)
        sorted_algorithms, sorted_accuracies = zip(*sorted_data)
        
        colors = ['#Gold', '#Silver', '#CD7F32', '#4ECDC4']  # Gold, Silver, Bronze, others
        bars = ax4.bar(sorted_algorithms, sorted_accuracies, 
                      color=colors[:len(sorted_algorithms)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, sorted_accuracies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Algorithm Performance Ranking', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Final Accuracy')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'research_questions_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function for enhanced thesis experiment"""
    print("Enhanced Master's Thesis: Comprehensive Federated Learning Analysis with Flower Algorithms")
    print("=" * 90)
    
    # Load configuration
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: Config file not found!")
        return
    
    # Initialize enhanced simulation
    simulation = EnhancedFLSimulation(config)
    
    # Run comprehensive experiment
    start_time = time.time()
    results, academic_analysis, research_analysis = simulation.run_comprehensive_experiment()
    end_time = time.time()
    
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
    print(f"Results available in results/enhanced_thesis_* directory")
    
    # Print top-level results
    if academic_analysis.get('algorithm_rankings'):
        print("\nAlgorithm Rankings:")
        for i, (algorithm, score_data) in enumerate(academic_analysis['algorithm_rankings'].items(), 1):
            print(f"{i}. {algorithm}: {score_data['mean_score']:.3f}")
    
    # Print research findings
    if research_analysis.get('fl_vs_centralized'):
        print("\nFL vs Centralized Learning Results:")
        for dataset, data in research_analysis['fl_vs_centralized'].items():
            print(f"  {dataset}: FL={data['fl_avg_accuracy']:.3f}, "
                  f"Centralized={data['centralized_accuracy']:.3f}, "
                  f"Gap={data['accuracy_gap']:.3f}")
    
    # Print device reliability findings
    if research_analysis.get('device_reliability'):
        print(f"\nDevice Reliability Analysis:")
        print(f"  Dropout events analyzed: {len(research_analysis['device_reliability'])}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Enhanced Comprehensive Federated Learning Simulation with Flower Algorithms
========================================================================
Advanced framework implementing SCAFFOLD, FedAdam, COOP algorithms from Flower
with comprehensive academic dashboards and research question analysis.

Author: Mehdi MOUALIM
Thesis: Comparative Analysis of Federated Learning Algorithms
"""

# Standard library imports
import os
import sys
import copy
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import OrderedDict

# Third-party imports
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveModel(nn.Module):
    """Enhanced adaptive neural network with additional metrics tracking"""
    def __init__(self, dataset_type='mnist', num_classes=10):
        super(AdaptiveModel, self).__init__()
        
        if dataset_type in ['mnist', 'femnist']:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif dataset_type in ['cifar10', 'cifar100']:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.log_softmax(x, dim=1)

class EnhancedFLSimulation:
    """Enhanced Federated Learning Simulation with Flower Algorithms"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Initialize algorithm-specific variables
        self.scaffold_global_c = None
        self.scaffold_client_c = None
        self.fedadam_m = None
        self.fedadam_v = None
        self.fedadam_tau = config.get('algorithm_params', {}).get('fedadam', {}).get('tau', 0.001)
        
        logger.info("Enhanced FL Simulation with Flower Algorithms")
        logger.info(f"Device: {self.device}")
        logger.info(f"Algorithms: {config['algorithms']}")
        logger.info(f"Datasets: {config['datasets']}")
        
    def setup_dataset(self, dataset_name: str):
        """Enhanced dataset setup with additional preprocessing"""
        logger.info(f"Setting up {dataset_name} dataset...")
        
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('data', train=False, transform=transform)
            num_classes = 10
            input_channels = 1
            
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('data', train=False, transform=transform)
            num_classes = 10
            input_channels = 3
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return train_dataset, test_dataset, num_classes, input_channels

    def create_federated_data(self, dataset, num_clients: int, distribution: str, beta: float = 0.5):
        """Create federated data with different heterogeneity levels"""
        logger.info(f"Creating {distribution} federated split (beta={beta}) for {num_clients} clients...")
        
        if distribution == 'iid':
            client_data = self._create_iid_split(dataset, num_clients)
        else:
            client_data = self._create_non_iid_split(dataset, num_clients, beta)
        
        # Calculate heterogeneity metrics
        heterogeneity_metrics = self.calculate_data_heterogeneity_metrics(client_data)
        
        return client_data, heterogeneity_metrics
    
    def _create_iid_split(self, dataset, num_clients):
        """Create IID data split"""
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split_size = len(indices) // num_clients
        
        client_data = []
        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_clients - 1 else len(indices)
            client_indices = indices[start_idx:end_idx]
            client_dataset = Subset(dataset, client_indices)
            client_data.append(client_dataset)
        
        return client_data
    
    def _create_non_iid_split(self, dataset, num_clients, beta):
        """Create Non-IID data split using Dirichlet distribution"""
        if hasattr(dataset, 'targets'):
            labels = dataset.targets if torch.is_tensor(dataset.targets) else torch.tensor(dataset.targets)
        else:
            labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(torch.unique(labels))
        
        # Group data by class
        class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            class_indices[label.item()].append(idx)
        
        # Distribute using Dirichlet
        client_indices = [[] for _ in range(num_clients)]
        
        for class_idx in range(num_classes):
            class_data = class_indices[class_idx]
            np.random.shuffle(class_data)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([beta] * num_clients)
            
            # Distribute class data
            start_idx = 0
            for client_idx in range(num_clients):
                end_idx = start_idx + int(proportions[client_idx] * len(class_data))
                end_idx = min(end_idx, len(class_data))
                client_indices[client_idx].extend(class_data[start_idx:end_idx])
                start_idx = end_idx
        
        # Create datasets
        client_data = []
        for indices in client_indices:
            if len(indices) > 0:
                client_dataset = Subset(dataset, indices)
                client_data.append(client_dataset)
        
        return client_data

    def calculate_data_heterogeneity_metrics(self, client_data):
        """Calculate comprehensive data heterogeneity metrics"""
        if not client_data:
            return {}
        
        # Extract label distributions per client
        client_distributions = []
        for client_dataset in client_data:
            if hasattr(client_dataset, 'dataset') and hasattr(client_dataset.dataset, 'targets'):
                # Handle Subset case
                indices = client_dataset.indices
                targets = client_dataset.dataset.targets
                client_targets = [targets[i] for i in indices]
            else:
                # Handle direct dataset case
                client_targets = [client_dataset[i][1] for i in range(len(client_dataset))]
            
            # Count label distribution
            unique, counts = np.unique(client_targets, return_counts=True)
            distribution = np.zeros(10)  # Assuming 10 classes
            for label, count in zip(unique, counts):
                if label < 10:
                    distribution[label] = count
            
            # Normalize to probabilities
            distribution = distribution / distribution.sum() if distribution.sum() > 0 else distribution
            client_distributions.append(distribution)
        
        if not client_distributions:
            return {}
        
        client_distributions = np.array(client_distributions)
        
        # Calculate various heterogeneity metrics
        metrics = {}
        
        # KL Divergence
        kl_divergences = []
        for i in range(len(client_distributions)):
            for j in range(i + 1, len(client_distributions)):
                epsilon = 1e-8
                p = client_distributions[i] + epsilon
                q = client_distributions[j] + epsilon
                kl = np.sum(p * np.log(p / q))
                kl_divergences.append(kl)
        metrics['kl_divergence'] = np.mean(kl_divergences) if kl_divergences else 0.0
        
        # Entropy-based measures
        client_entropies = []
        for dist in client_distributions:
            entropy = -np.sum(dist * np.log(dist + 1e-8))
            client_entropies.append(entropy)
        metrics['avg_client_entropy'] = np.mean(client_entropies)
        metrics['entropy_variance'] = np.var(client_entropies)
        
        return metrics

    def run_centralized_baseline(self, dataset_name: str):
        """Run centralized learning baseline for comparison"""
        logger.info(f"Running centralized baseline for {dataset_name}")
        
        train_dataset, test_dataset, num_classes, _ = self.setup_dataset(dataset_name)
        
        # Create centralized data loader
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'] * 4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # Initialize model
        model = AdaptiveModel(dataset_name, num_classes).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.NLLLoss()
        
        # Training metrics
        centralized_metrics = {
            'rounds': [],
            'test_accuracy': [],
            'test_loss': [],
            'train_loss': [],
            'round_times': []
        }
        
        # Train for equivalent epochs as federated setting
        total_rounds = self.config['num_rounds']
        
        for round_num in range(total_rounds):
            round_start = time.time()
            model.train()
            
            epoch_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Evaluation
            test_acc, test_loss = self._evaluate_model(model, test_loader)
            
            # Store metrics
            round_time = time.time() - round_start
            centralized_metrics['rounds'].append(round_num + 1)
            centralized_metrics['test_accuracy'].append(test_acc)
            centralized_metrics['test_loss'].append(test_loss)
            centralized_metrics['train_loss'].append(epoch_loss / len(train_loader))
            centralized_metrics['round_times'].append(round_time)
            
            if round_num % 10 == 0:
                logger.info(f"Centralized Round {round_num + 1}: Acc={test_acc:.4f}, Loss={test_loss:.4f}")
        
        # Store final metrics
        centralized_metrics['final_accuracy'] = centralized_metrics['test_accuracy'][-1] if centralized_metrics['test_accuracy'] else 0.0
        centralized_metrics['final_loss'] = centralized_metrics['test_loss'][-1] if centralized_metrics['test_loss'] else 0.0
        centralized_metrics['convergence_round'] = len(centralized_metrics['rounds'])
        centralized_metrics['total_time'] = sum(centralized_metrics['round_times'])
        
        return centralized_metrics

    def run_enhanced_algorithm(self, algorithm: str, dataset_name: str, distribution: str, beta: float):
        """Enhanced algorithm execution with comprehensive metrics"""
        logger.info(f"Running enhanced {algorithm} on {dataset_name} ({distribution}, beta={beta})")
        
        # Setup data
        train_dataset, test_dataset, num_classes, input_channels = self.setup_dataset(dataset_name)
        client_data, heterogeneity_metrics = self.create_federated_data(
            train_dataset, self.config['num_clients'], distribution, beta
        )
        
        # Create data loaders
        client_loaders = [DataLoader(data, batch_size=self.config['batch_size'], shuffle=True) 
                         for data in client_data]
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # Initialize model
        global_model = AdaptiveModel(dataset_name, num_classes).to(self.device)
        
        # Initialize algorithm-specific variables
        self._initialize_algorithm_variables(global_model, algorithm, len(client_loaders))
        
        # Enhanced training metrics
        metrics = {
            'algorithm': algorithm,
            'dataset': dataset_name,
            'distribution': distribution,
            'beta': beta,
            'rounds': [],
            'test_accuracy': [],
            'test_loss': [],
            'train_loss': [],
            'communication_cost': [],
            'model_drift': [],
            'client_fairness': [],
            'convergence_rate': [],
            'privacy_budget': [],
            'round_times': [],
            'gradient_diversity': [],
            'client_participation': [],
            'model_stability': [],
            'heterogeneity_metrics': heterogeneity_metrics,
            'dropout_analysis': {},
            'gradient_norms': [],
            'client_accuracy_variance': [],
            'consensus_metrics': []
        }
        
        total_comm_cost = 0
        model_parameters = sum(p.numel() for p in global_model.parameters())
        
        # Training loop with enhanced analysis
        for round_num in range(self.config['num_rounds']):
            round_start = time.time()
            
            # Simulate device dropouts if enabled
            if self.config.get('robustness_testing', {}).get('dropout_simulation', False):
                dropout_rate = self._get_dropout_rate(round_num)
                active_clients, dropped_clients = self.simulate_device_dropouts(client_loaders, dropout_rate)
                if dropout_rate > 0:
                    metrics['dropout_analysis'][round_num] = {
                        'dropout_rate': dropout_rate,
                        'active_clients': len(active_clients),
                        'dropped_clients': len(dropped_clients)
                    }
            else:
                active_clients = list(range(len(client_loaders)))
                dropped_clients = []
            
            # Client selection from active clients
            selected_clients = np.random.choice(
                active_clients,
                size=min(self.config['clients_per_round'], len(active_clients)),
                replace=False
            )
            
            # Client training with gradient tracking
            client_weights = []
            client_samples = []
            client_losses = []
            client_accuracies = []
            client_gradients = []
            
            for client_id in selected_clients:
                weights, samples, loss, acc, gradients = self._enhanced_client_training(
                    client_id, global_model, client_loaders[client_id], algorithm, round_num
                )
                client_weights.append(weights)
                client_samples.append(samples)
                client_losses.append(loss)
                client_accuracies.append(acc)
                client_gradients.append(gradients)
                
                # Communication cost (upload + download)
                total_comm_cost += model_parameters * 8
            
            # Analyze gradient diversity
            grad_similarity, grad_std, grad_magnitude_var = self.analyze_gradient_diversity(client_gradients)
            
            # Server aggregation
            global_model = self._server_aggregation(global_model, client_weights, client_samples, algorithm, round_num)
            
            # Evaluation
            test_acc, test_loss = self._evaluate_model(global_model, test_loader)
            train_loss = np.mean(client_losses) if client_losses else 0.0
            
            # Enhanced metrics calculation
            model_drift = self._calculate_model_drift(client_weights, global_model.state_dict())
            fairness_score = self._calculate_fairness(client_accuracies)
            convergence_rate = self._calculate_convergence_rate(metrics['test_accuracy'], test_acc)
            privacy_budget = self._calculate_privacy_budget(round_num, self.config.get('dp_epsilon', 1.0))
            
            # Model stability analysis
            stability_score = self._calculate_model_stability(client_weights)
            
            # Client participation analysis
            participation_rate = len(selected_clients) / len(client_loaders)
            
            # Consensus metrics
            consensus_score = self._calculate_consensus_score(client_accuracies, test_acc)
            
            # Store enhanced metrics
            round_time = time.time() - round_start
            metrics['rounds'].append(round_num + 1)
            metrics['test_accuracy'].append(test_acc)
            metrics['test_loss'].append(test_loss)
            metrics['train_loss'].append(train_loss)
            metrics['communication_cost'].append(total_comm_cost)
            metrics['model_drift'].append(model_drift)
            metrics['client_fairness'].append(fairness_score)
            metrics['convergence_rate'].append(convergence_rate)
            metrics['privacy_budget'].append(privacy_budget)
            metrics['round_times'].append(round_time)
            metrics['gradient_diversity'].append(grad_similarity)
            metrics['client_participation'].append(participation_rate)
            metrics['model_stability'].append(stability_score)
            metrics['client_accuracy_variance'].append(np.var(client_accuracies) if client_accuracies else 0.0)
            metrics['consensus_metrics'].append(consensus_score)
            
            if round_num % 10 == 0:
                logger.info(f"Round {round_num + 1}: Acc={test_acc:.4f}, Loss={test_loss:.4f}, "
                           f"Drift={model_drift:.4f}, Fairness={fairness_score:.4f}")
            
            # Early stopping
            if test_acc >= self.config.get('target_accuracy', 0.95):
                logger.info(f"Target accuracy reached at round {round_num + 1}")
                break
        
        # Calculate final comprehensive metrics
        metrics.update(self._calculate_final_metrics(metrics))
        
        return metrics

    def _initialize_algorithm_variables(self, model, algorithm, num_clients):
        """Initialize algorithm-specific variables"""
        if algorithm.lower() == 'scaffold':
            # Initialize SCAFFOLD control variates
            self.scaffold_global_c = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            self.scaffold_client_c = [{name: torch.zeros_like(param) for name, param in model.named_parameters()} 
                                    for _ in range(num_clients)]
        elif algorithm.lower() == 'fedadam':
            # Initialize FedAdam momentum variables
            self.fedadam_m = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            self.fedadam_v = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def _get_dropout_rate(self, round_num):
        """Get dropout rate for current round"""
        dropout_rates = self.config.get('robustness_testing', {}).get('dropout_rates', [0.0, 0.1, 0.2, 0.3])
        if round_num % 20 == 0 and round_num > 0:
            return dropout_rates[min(round_num // 20, len(dropout_rates) - 1)]
        return 0.0

    def simulate_device_dropouts(self, client_loaders, dropout_rate=0.3):
        """Simulate device dropouts and irregular updates"""
        num_clients = len(client_loaders)
        dropout_clients = np.random.choice(
            num_clients, 
            size=int(num_clients * dropout_rate), 
            replace=False
        )
        
        active_clients = [i for i in range(num_clients) if i not in dropout_clients]
        return active_clients, dropout_clients

    def _enhanced_client_training(self, client_id: int, global_model: nn.Module, 
                                client_loader: DataLoader, algorithm: str, round_num: int):
        """Enhanced client training with algorithm-specific implementations"""
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        # Algorithm-specific setup
        if algorithm.lower() == 'fedprox':
            optimizer = optim.SGD(local_model.parameters(), lr=self.config['learning_rate'])
            mu = self.config.get('algorithm_params', {}).get('fedprox', {}).get('mu', 0.01)
            global_params = {name: param.clone() for name, param in global_model.named_parameters()}
        elif algorithm.lower() == 'fedadam':
            optimizer = optim.Adam(local_model.parameters(), lr=self.config['learning_rate'])
        elif algorithm.lower() == 'scaffold':
            optimizer = optim.SGD(local_model.parameters(), lr=self.config['learning_rate'])
            # SCAFFOLD control variates
            if self.scaffold_client_c and client_id < len(self.scaffold_client_c):
                client_c = self.scaffold_client_c[client_id]
                global_c = self.scaffold_global_c
            else:
                client_c = {name: torch.zeros_like(param) for name, param in local_model.named_parameters()}
                global_c = {name: torch.zeros_like(param) for name, param in local_model.named_parameters()}
        elif algorithm.lower() == 'coop':
            optimizer = optim.SGD(local_model.parameters(), lr=self.config['learning_rate'])
            alpha = self.config.get('algorithm_params', {}).get('coop', {}).get('alpha', 0.1)
            beta = self.config.get('algorithm_params', {}).get('coop', {}).get('beta', 0.1)
        else:  # FedAvg
            optimizer = optim.SGD(local_model.parameters(), lr=self.config['learning_rate'])
        
        criterion = nn.NLLLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        gradients = {}
        
        # Store initial parameters for SCAFFOLD
        if algorithm.lower() == 'scaffold':
            initial_params = {name: param.clone() for name, param in local_model.named_parameters()}
        
        for epoch in range(self.config['local_epochs']):
            for data, target in client_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                
                # Algorithm-specific loss modifications
                if algorithm.lower() == 'fedprox':
                    # Add proximal term
                    proximal_term = 0.0
                    for name, param in local_model.named_parameters():
                        if name in global_params:
                            proximal_term += torch.norm(param - global_params[name]) ** 2
                    loss += (mu / 2) * proximal_term
                elif algorithm.lower() == 'coop':
                    # Add cooperation regularization
                    coop_term = 0.0
                    for name, param in local_model.named_parameters():
                        if name in global_params:
                            coop_term += torch.norm(param - global_params[name]) ** 2
                    loss += (beta / 2) * coop_term
                
                loss.backward()
                
                # SCAFFOLD gradient correction
                if algorithm.lower() == 'scaffold':
                    for name, param in local_model.named_parameters():
                        if param.grad is not None and name in client_c and name in global_c:
                            param.grad += global_c[name] - client_c[name]
                
                # Store gradients for analysis
                for name, param in local_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone()
                
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
        
        # Update SCAFFOLD control variates
        if algorithm.lower() == 'scaffold':
            self._update_scaffold_control_variates(client_id, initial_params, local_model, client_c)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return local_model.state_dict(), total_samples, avg_loss, accuracy, gradients

    def _update_scaffold_control_variates(self, client_id, initial_params, local_model, client_c):
        """Update SCAFFOLD control variates"""
        lr = self.config['learning_rate']
        local_epochs = self.config['local_epochs']
        
        for name, param in local_model.named_parameters():
            if name in initial_params and name in client_c:
                delta = initial_params[name] - param
                client_c[name] = client_c[name] - delta / (local_epochs * lr)

    def _server_aggregation(self, global_model: nn.Module, client_weights: List, 
                          client_samples: List, algorithm: str, round_num: int):
        """Server-side aggregation with algorithm-specific methods"""
        if algorithm.lower() == 'fedavg':
            return self._fedavg_aggregation(global_model, client_weights, client_samples)
        elif algorithm.lower() == 'fedprox':
            return self._fedavg_aggregation(global_model, client_weights, client_samples)
        elif algorithm.lower() == 'fedadam':
            return self._fedadam_aggregation(global_model, client_weights, client_samples, round_num)
        elif algorithm.lower() == 'scaffold':
            return self._scaffold_aggregation(global_model, client_weights, client_samples)
        elif algorithm.lower() == 'coop':
            return self._coop_aggregation(global_model, client_weights, client_samples)
        else:
            return self._fedavg_aggregation(global_model, client_weights, client_samples)

    def _fedavg_aggregation(self, global_model, client_weights, client_samples):
        """FedAvg weighted averaging"""
        total_samples = sum(client_samples)
        aggregated_weights = {}
        
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
        
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight_factor = num_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight_factor * client_weight[key]
        
        global_model.load_state_dict(aggregated_weights)
        return global_model

    def _fedadam_aggregation(self, global_model, client_weights, client_samples, round_num):
        """FedAdam server aggregation with adaptive optimization"""
        # First perform FedAvg aggregation
        global_model = self._fedavg_aggregation(global_model, client_weights, client_samples)
        
        # Apply FedAdam momentum and adaptive learning
        beta1 = self.config.get('algorithm_params', {}).get('fedadam', {}).get('beta1', 0.9)
        beta2 = self.config.get('algorithm_params', {}).get('fedadam', {}).get('beta2', 0.999)
        tau = self.fedadam_tau
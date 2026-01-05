#python ensemble.py --data combined_data.csv --target PP
from pc_bootstrap import bootstrap_pc_algorithm
from ges_bootstrap import bootstrap_dag
from notears_bootstrap import bootstrap_notears

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm

class EnsembleCausalDiscovery:
    
    def __init__(self, data: pd.DataFrame, target_var: str):
        
        self.data = data
        self.target_var = target_var
        self.target_idx = list(data.columns).index(target_var)
        self.variable_names = list(data.columns)
        
        # Define threshold settings for different levels
        self.threshold_settings = {
            'conservative': {
                'threshold': 0.45,  
                'color': '#1a5f7a'  
            },
            'moderate': {
                'threshold': 0.35,  
                'color': '#2e8b57'  
            },
            'liberal': {
                'threshold': 0.25,  
                'color': '#8b4513'  
            }
        }

    def run_algorithm(self, 
                     n_bootstraps: int = 500,
                     pc_alpha: float = 0.05,
                     pc_confidence: float = 0.3,
                     ges_threshold: float = 0.3,
                     notears_threshold: float = 0.3) -> Dict[str, nx.DiGraph]:
       
        print("Running ensemble causal discovery algorithms...")
        
        # Run all algorithms
        graphs = []
        edge_info_list = []
        
        # Run PC algorithm
        print("\nRunning PC algorithm...")
        try:
            edges_info, dag1, _ = bootstrap_pc_algorithm(
                self.data, 
                self.target_var,
                n_bootstraps=n_bootstraps,
                alpha=pc_alpha,
                confidence_threshold=pc_confidence
            )
            graphs.append(dag1)
            edge_info_list.append(edges_info)
            print("PC algorithm completed successfully")
        except Exception as e:
            print(f"Warning: PC algorithm failed: {str(e)}")
            graphs.append(nx.DiGraph())
            edge_info_list.append({})

        # Run GES algorithm
        print("\nRunning GES algorithm...")
        try:
            final_dag2, edge_frequencies, _ = bootstrap_dag(
                self.data.values,
                self.variable_names,
                self.target_var,
                n_bootstrap=n_bootstraps,
                threshold=ges_threshold
            )
            dag2 = nx.DiGraph()
            for i, source in enumerate(self.variable_names):
                for j, target in enumerate(self.variable_names):
                    if final_dag2[i, j] == 1:
                        dag2.add_edge(source, target, weight=edge_frequencies[i, j])
            graphs.append(dag2)
            print("GES algorithm completed successfully")
        except Exception as e:
            print(f"Warning: GES algorithm failed: {str(e)}")
            graphs.append(nx.DiGraph())

        # Run NOTEARS algorithm
        print("\nRunning NOTEARS algorithm...")
        try:
            W_freq, W_mean, W_std = bootstrap_notears(
                self.data.values,
                n_bootstrap=n_bootstraps,
                target_var_idx=self.target_idx
            )
            dag3 = nx.DiGraph()
            for i, source in enumerate(self.variable_names):
                for j, target in enumerate(self.variable_names):
                    if W_freq[i, j] >= notears_threshold:
                        dag3.add_edge(source, target,
                                    weight=W_mean[i, j],
                                    frequency=W_freq[i, j],
                                    std=W_std[i, j])
            graphs.append(dag3)
            print("NOTEARS algorithm completed successfully")
        except Exception as e:
            print(f"Warning: NOTEARS algorithm failed: {str(e)}")
            graphs.append(nx.DiGraph())

        # Compute agreement scores
        agreement_scores = self._compute_edge_agreement(graphs)
        
        # Create ensemble DAGs for different thresholds
        ensemble_dags = {}
        for setting, params in self.threshold_settings.items():
            ensemble_dags[setting] = self._create_ensemble_dag(
                agreement_scores,
                params['threshold']
            )
        
        # Save results
        self._save_analysis_results(ensemble_dags, agreement_scores, graphs, edge_info_list)
        
        return ensemble_dags

    def _compute_edge_agreement(self, graphs: List[nx.DiGraph]) -> Dict:
        
        edge_votes = {}
        edge_weights = {}
        
        # Count votes and collect weights for each edge
        for graph in graphs:
            for edge in graph.edges(data=True):
                source, target = edge[0], edge[1]
                key = (source, target)
                
                # Count votes
                edge_votes[key] = edge_votes.get(key, 0) + 1
                
                # Collect weights
                weight = edge[2].get('weight', 
                                   edge[2].get('frequency',
                                   edge[2].get('confidence', 1.0)))
                if key not in edge_weights:
                    edge_weights[key] = []
                edge_weights[key].append(float(weight))
        
        # Compute final agreement scores
        agreement_scores = {}
        for edge, votes in edge_votes.items():
            weights = edge_weights[edge]
            agreement_scores[edge] = {
                'vote_ratio': votes / len(graphs),
                'mean_weight': np.mean(weights),
                'std_weight': np.std(weights) if len(weights) > 1 else 0.0
            }
        
        return agreement_scores

    def _create_ensemble_dag(self, 
                           agreement_scores: Dict,
                           threshold: float) -> nx.DiGraph:
       
        ensemble_dag = nx.DiGraph()
        
        # Add all nodes
        for col in self.data.columns:
            ensemble_dag.add_node(col)
        
        # Add edges that meet the combined threshold criteria
        for edge, scores in agreement_scores.items():
            combined_score = scores['vote_ratio'] * scores['mean_weight']
            if combined_score >= threshold:
                source, target = edge
                ensemble_dag.add_edge(source, target, **scores)
        
        # Ensure DAG properties
        self._ensure_dag_properties(ensemble_dag)
        
        return ensemble_dag

    def _ensure_dag_properties(self, dag: nx.DiGraph):
        
        # Remove cycles
        while not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            if not cycles:
                break
            
            cycle = cycles[0]
            min_score = float('inf')
            edge_to_remove = None
            
            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]
                edge_data = dag.get_edge_data(source, target)
                if edge_data:
                    score = edge_data['vote_ratio'] * edge_data['mean_weight']
                    if score < min_score:
                        min_score = score
                        edge_to_remove = (source, target)
            
            if edge_to_remove:
                dag.remove_edge(*edge_to_remove)
        
        # Connect leaf nodes to target
        leaf_nodes = [node for node in dag.nodes() 
                     if node != self.target_var and dag.out_degree(node) == 0]
        
        for leaf in leaf_nodes:
            if not dag.has_edge(leaf, self.target_var):
                corr = abs(np.corrcoef(self.data[leaf], 
                                     self.data[self.target_var])[0, 1])
                dag.add_edge(leaf, self.target_var,
                            vote_ratio=0.5,
                            mean_weight=corr,
                            std_weight=0.0)

    def _save_analysis_results(self,
                             ensemble_dags: Dict[str, nx.DiGraph],
                             agreement_scores: Dict,
                             graphs: List[nx.DiGraph],
                             edge_info_list: List[Dict]):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"causal_analysis_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save basic information
        info = {
            "nodes": list(self.data.columns),
            "target_variable": self.target_var,
            "timestamp": timestamp,
            "threshold_settings": {
                setting: {
                    "edges": len(dag.edges()),
                    "threshold": self.threshold_settings[setting]['threshold']
                }
                for setting, dag in ensemble_dags.items()
            }
        }
        with open(f"{output_dir}/basic_info.json", "w") as f:
            json.dump(info, f, indent=4)
        
        # Create visualizations for each threshold setting
        for setting, dag in ensemble_dags.items():
            self._create_dag_visualization(
                dag,
                f"{output_dir}/causal_graph_{setting}",
                self.threshold_settings[setting]['color'],
                setting
            )
        
        # Generate comparative report
        self._generate_comparative_report(ensemble_dags, output_dir)
        
        # Create additional analysis plots
        self._create_analysis_plots(ensemble_dags, output_dir)

    def _create_dag_visualization(self, 
                                dag: nx.DiGraph, 
                                filename: str, 
                                color: str, 
        
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(dag, k=1, iterations=50)
        
        # Draw nodes
        node_colors = ['lightpink' if node == self.target_var else 'lightblue' 
                      for node in dag.nodes()]
        nx.draw_networkx_nodes(dag, pos, 
                             node_color=node_colors,
                             node_size=2000,
                             edgecolors='black')
        
        # Draw node labels
        nx.draw_networkx_labels(dag, pos, font_size=10)
        
        # Draw edges with varying width and add edge labels
        edge_weights = []
        edge_labels = {}
        
        for source, target, data in dag.edges(data=True):
            strength = data['vote_ratio'] * data['mean_weight']
            edge_weights.append(1 + 2 * strength)
            edge_labels[(source, target)] = (f"VR: {data['vote_ratio']:.2f}\n"
                                           f"W: {data['mean_weight']:.2f}")
        
        # Draw edges
        nx.draw_networkx_edges(dag, pos,
                      width=edge_weights,
                      edge_color=color,
                      arrowsize=20,
                      arrowstyle='->',
                      connectionstyle='arc3, rad=0.1')
        
        # Add edge labels
        nx.draw_networkx_edge_labels(dag, pos,
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Add title
        plt.title(f'{setting.capitalize()} Threshold Setting\n'
                 f'(threshold: {self.threshold_settings[setting]["threshold"]:.2f})',
                 pad=20, fontsize=14)
        
        # Adjust layout
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()


    def _create_analysis_plots(self, 
                             ensemble_dags: Dict[str, nx.DiGraph],
       
        # Plot edge count comparison
        plt.figure(figsize=(10, 6))
        settings = list(ensemble_dags.keys())
        edge_counts = [len(dag.edges()) for dag in ensemble_dags.values()]
        
        plt.bar(settings, edge_counts)
        plt.title('Edge Count Comparison Across Threshold Settings')
        plt.ylabel('Number of Edges')
        plt.savefig(f"{output_dir}/edge_count_comparison.png")
        plt.close()
        
        # Plot edge strength distributions
        plt.figure(figsize=(12, 6))
        for setting, dag in ensemble_dags.items():
            strengths = [d['vote_ratio'] * d['mean_weight'] 
                        for _, _, d in dag.edges(data=True)]
            if strengths:
                sns.kdeplot(data=strengths, label=setting)
        
        plt.title('Edge Strength Distributions')
        plt.xlabel('Edge Strength (Vote Ratio × Mean Weight)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f"{output_dir}/edge_strength_distributions.png")
        plt.close()

    def _generate_comparative_report(self, 
                                   ensemble_dags: Dict[str, nx.DiGraph], 
       
        with open(f"{output_dir}/comparative_analysis.txt", "w") as f:
            f.write("Comparative Analysis of Different Threshold Settings\n")
            f.write("=" * 50 + "\n\n")
            
            for setting, dag in ensemble_dags.items():
                f.write(f"\n{setting.capitalize()} Setting Analysis\n")
                f.write("-" * 30 + "\n")
                
                # Basic statistics
                f.write(f"Threshold parameters:\n")
                f.write(f"  - Combined threshold: {self.threshold_settings[setting]['threshold']}\n")
                
                # Basic statistics
                f.write(f"\nNetwork Statistics:\n")
                f.write(f"  - Total edges: {len(dag.edges())}\n")
                f.write(f"  - Direct connections to target: {len(list(dag.predecessors(self.target_var)))}\n")
                
                # Analyzing direct causes
                direct_causes = list(dag.predecessors(self.target_var))
                if direct_causes:
                    f.write("\nDirect causes of target variable:\n")
                    for cause in direct_causes:
                        edge_data = dag.get_edge_data(cause, self.target_var)
                        strength = edge_data['vote_ratio'] * edge_data['mean_weight']
                        f.write(f"  - {cause} (strength: {strength:.3f})\n")
                
                # List strongest relationships
                f.write("\nStrongest causal relationships:\n")
                edges = [(s, t, d) for s, t, d in dag.edges(data=True)]
                edges.sort(key=lambda x: x[2]['vote_ratio'] * x[2]['mean_weight'], 
                         reverse=True)
                
                for s, t, d in edges[:5]:
                    strength = d['vote_ratio'] * d['mean_weight']
                    f.write(f"  {s} -> {t} (strength: {strength:.3f}, "
                           f"vote ratio: {d['vote_ratio']:.2f}, "
                           f"weight: {d['mean_weight']:.2f})\n")
                
                # Detailed Path Analysis
                f.write("\nDetailed Path Analysis:\n")
                f.write("-" * 20 + "\n")
                
                # Collect all paths to target variable
                all_paths = []
                for node in dag.nodes():
                    if node != self.target_var:
                        try:
                            paths = list(nx.all_simple_paths(dag, node, self.target_var))
                            if paths:
                                # Calculate path strength for each path
                                for path in paths:
                                    path_strength = self._calculate_path_strength(dag, path)
                                    all_paths.append({
                                        'start': node,
                                        'path': path,
                                        'length': len(path),
                                        'strength': path_strength
                                    })
                        except nx.NetworkXNoPath:
                            continue
                
                # Sort paths by strength
                all_paths.sort(key=lambda x: x['strength'], reverse=True)
                
                # Output path information for each starting node
                for node in sorted(set(p['start'] for p in all_paths)):
                    node_paths = [p for p in all_paths if p['start'] == node]
                    f.write(f"\nPaths from {node} to {self.target_var}:\n")
                    
                    for idx, path_info in enumerate(node_paths, 1):
                        path = path_info['path']
                        path_str = " -> ".join(path)
                        f.write(f"  Path {idx}: {path_str}\n")
                        f.write(f"    Length: {path_info['length']}\n")
                        f.write(f"    Strength: {path_info['strength']:.3f}\n")
                        
                        # Output edge details for each path
                        f.write("    Edge details:\n")
                        for i in range(len(path)-1):
                            source, target = path[i], path[i+1]
                            edge_data = dag.get_edge_data(source, target)
                            f.write(f"      {source} -> {target}:\n")
                            f.write(f"        Vote ratio: {edge_data['vote_ratio']:.3f}\n")
                            f.write(f"        Mean weight: {edge_data['mean_weight']:.3f}\n")
                            f.write(f"        Std weight: {edge_data['std_weight']:.3f}\n")
                    
                    f.write("\n" + "=" * 50 + "\n")

    def _calculate_path_strength(self, dag: nx.DiGraph, path: List[str]) -> float:
       
        # Calculate path strength (using the product of edge strengths)
        strength = 1.0
        for i in range(len(path)-1):
            source, target = path[i], path[i+1]
            edge_data = dag.get_edge_data(source, target)
            edge_strength = edge_data['vote_ratio'] * edge_data['mean_weight']
            strength *= edge_strength
        return strength

def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Causal Discovery Analysis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data CSV file')
    parser.add_argument('--target', type=str, required=True,
                       help='Name of target variable')
    parser.add_argument('--bootstraps', type=int, default=500,
                       help='Number of bootstrap iterations')
    parser.add_argument('--pc-alpha', type=float, default=0.05,
                       help='Significance level for PC algorithm')
    parser.add_argument('--pc-confidence', type=float, default=0.3,
                       help='Confidence threshold for PC algorithm')
    parser.add_argument('--ges-threshold', type=float, default=0.3,
                       help='Edge frequency threshold for GES')
    parser.add_argument('--notears-threshold', type=float, default=0.3,
                       help='Edge frequency threshold for NOTEARS')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory name')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    try:
        data = pd.read_csv(args.data)
        print(f"Loaded data with {data.shape[0]} samples and {data.shape[1]} variables")
        
        # Basic data checks
        if data.isnull().values.any():
            print("\nWarning: Dataset contains missing values")
            
        if (data.std() == 0).any():
            print("\nWarning: Some variables have zero variance")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Verify target variable exists
    if args.target not in data.columns:
        print(f"Error: Target variable '{args.target}' not found in data")
        print(f"Available variables: {', '.join(data.columns)}")
        return
    
    # Run analysis
    try:
        print("\nInitializing ensemble causal discovery...")
        ecd = EnsembleCausalDiscovery(data, args.target)
        
        ensemble_dags = ecd.run_algorithm(
            n_bootstraps=args.bootstraps,
            pc_alpha=args.pc_alpha,
            pc_confidence=args.pc_confidence,
            ges_threshold=args.ges_threshold,
            notears_threshold=args.notears_threshold
        )
        
        print("\nAnalysis completed successfully!")
        print("\nResults summary:")
        for setting, dag in ensemble_dags.items():
            print(f"\n{setting.capitalize()} threshold setting:")
            print(f"- Found {len(dag.edges())} causal relationships")
            print(f"- Direct causes of {args.target}: "
                  f"{len(list(dag.predecessors(args.target)))}")
        
        print("\nDetailed results have been saved to the 'causal_analysis_results_<timestamp>' directory")
        print("\nGenerated files include:")
        print("- Basic information (basic_info.json)")
        print("- Causal graphs for each threshold setting (causal_graph_*.png)")
        print("- Comparative analysis report (comparative_analysis.txt)")
        print("- Analysis plots (edge_count_comparison.png, edge_strength_distributions.png)")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

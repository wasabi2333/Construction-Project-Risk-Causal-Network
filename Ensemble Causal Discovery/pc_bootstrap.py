import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Tuple

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.cit import kci
from causallearn.utils.cit import gsq
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode


data = pd.read_csv('combined_data.csv')

def estimate_edge_strength(data: np.ndarray, i: int, j: int) -> float:
    
    try:
        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
        return abs(corr)
    except:
        return 0

def analyze_edge_distribution(edges_info: Dict) -> Dict:
   
    if not edges_info:
        return {
            'conservative': {'confidence': 0.04, 'max_edges': 5},
            'moderate': {'confidence': 0.03, 'max_edges': 10},
            'liberal': {'confidence': 0.01, 'max_edges': 15}
        }
    
    importance_scores = [info['importance'] for info in edges_info.values()]
    confidence_scores = [info['confidence'] for info in edges_info.values()]
    
    total_edges = len(edges_info)
    
    suggestions = {
        'conservative': {
            'confidence': np.percentile(confidence_scores, 75) if confidence_scores else 0.03,
            'max_edges': max(5, int(total_edges * 0.1))
        },
        'moderate': {
            'confidence': np.percentile(confidence_scores, 50) if confidence_scores else 0.02,
            'max_edges': max(10, int(total_edges * 0.2))
        },
        'liberal': {
            'confidence': np.percentile(confidence_scores, 25) if confidence_scores else 0.01,
            'max_edges': max(15, int(total_edges * 0.3))
        }
    }
    
    return suggestions


def check_data_validity(data: np.ndarray) -> Tuple[bool, str]:
   
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False, "Data contains NaN or infinite values"
    
   
    std_values = np.std(data, axis=0)
    constant_cols = np.where(std_values < 1e-10)[0]  
    if len(constant_cols) > 0:
        return False, f"Near-constant columns found at indices: {constant_cols}"
    
   
    corr = np.corrcoef(data.T)
    mask = np.triu(np.ones_like(corr), k=1).astype(bool)
    if np.any(np.abs(corr[mask]) > 0.99999): 
        return False, "Near-perfect correlations found between variables"
        
   
    try:
        cond = np.linalg.cond(corr)
        if cond > 1e20:  
            return False, f"Correlation matrix is severely ill-conditioned"
    except np.linalg.LinAlgError:
        return False, "Correlation matrix is singular"
        
    return True, "Data is valid"

def estimate_skeleton_causallearn(
    bootstrap_data: np.ndarray,
    alpha: float,
    target_var: str,
    column_names: list
) -> np.ndarray:
    
    
    is_valid, message = check_data_validity(bootstrap_data)
    if not is_valid:
        raise ValueError(f"Data validation failed: {message}")
    
   
    n_vars = bootstrap_data.shape[1]
    nodes = [GraphNode(name) for name in column_names]
    target_idx = column_names.index(target_var)
    
   
    bk = BackgroundKnowledge()
    for i in range(n_vars):
        if i != target_idx:
            bk.add_forbidden_by_node(nodes[target_idx], nodes[i])
    
    
    try:
        result = pc(
            data=bootstrap_data,
            alpha=alpha,
            indep_test=fisherz, 
            stable=True,
            uc_rule=0,
            background_knowledge=bk,
            max_condition_set_size=0, 
            max_path_length=5,  
            verbose=False
        )
    except Exception as e:
        print(f"PC algorithm failed: {str(e)}")
        return np.zeros((n_vars, n_vars))
    
   
    adjacency = np.zeros((n_vars, n_vars))
    graph_nodes = result.G.get_nodes()
    
   
    for i in range(n_vars):
        if i != target_idx:
            node_i = graph_nodes[i]
            node_target = graph_nodes[target_idx]
            edge = result.G.get_edge(node_i, node_target)
            
            if edge is not None:
                
                if edge.get_endpoint1() == 2 and edge.get_endpoint2() == 1:  
                    adjacency[i, target_idx] = 1
    
    
    for i in range(n_vars):
        node_i = graph_nodes[i]
        for j in range(n_vars):
            if i != j and (i != target_idx and j != target_idx):
                node_j = graph_nodes[j]
                edge = result.G.get_edge(node_i, node_j)
                
                if edge is not None:
                    if edge.get_endpoint1() == 2 and edge.get_endpoint2() == 1:
                        adjacency[i, j] = 1
                    elif edge.get_endpoint1() == 1 and edge.get_endpoint2() == 2:
                        adjacency[j, i] = 1
    
    
    if not any(adjacency[:, target_idx]):
        
        correlations = np.abs([np.corrcoef(bootstrap_data[:, i], 
                                         bootstrap_data[:, target_idx])[0,1] 
                             for i in range(n_vars) if i != target_idx])
        if len(correlations) > 0:
            strongest_predictor = np.argmax(correlations)
            adjacency[strongest_predictor, target_idx] = 1
    
    return adjacency


print("\nChecking data properties:")
print("Data shape:", data.shape)
print("\nVariable standard deviations:")
print(data.std())
print("\nCorrelation matrix:")
print(data.corr().round(3))


is_valid, message = check_data_validity(data.values)
print("\nData validation result:", message)


def bootstrap_pc_algorithm(
    data: pd.DataFrame, 
    target_var: str,
    n_bootstraps: int = 500, 
    alpha: float = 0.2,
    confidence_threshold: float = 0.01,
    max_edges: int = 15,
    selection_mode: str = 'liberal'
):
    
    n_samples, n_vars = data.shape
    column_names = list(data.columns)
    
    
    edge_counts = {}
    edge_strengths = {}
    
   
    with tqdm(total=n_bootstraps, desc="Running bootstrap iterations") as pbar:
        for _ in range(n_bootstraps):
            try:
               
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_data = data.values[bootstrap_indices]
                
               
                adjacency = estimate_skeleton_causallearn(
                    bootstrap_data=bootstrap_data,
                    alpha=alpha,
                    target_var=target_var,
                    column_names=column_names
                )
                
                
                for i in range(n_vars):
                    for j in range(n_vars):
                        if adjacency[i, j] == 1:
                            edge = (column_names[i], column_names[j])
                            edge_counts[edge] = edge_counts.get(edge, 0) + 1
                            
                            strength = estimate_edge_strength(bootstrap_data, i, j)
                            if edge not in edge_strengths:
                                edge_strengths[edge] = []
                            edge_strengths[edge].append(strength)
            
            except Exception as e:
                print(f"Warning in bootstrap iteration: {str(e)}")
                continue
            
            pbar.update(1)
    
    
    edges_info = {}
    for edge, count in edge_counts.items():
        confidence = count / n_bootstraps
        mean_strength = np.mean(edge_strengths.get(edge, [0]))
        
        edges_info[edge] = {
            'confidence': confidence,
            'effect_size': mean_strength,
            'importance': confidence * mean_strength
        }
    
    
    suggestions = analyze_edge_distribution(edges_info)
    
   
    if selection_mode == 'auto':
        selection_mode = 'liberal'
    
    if confidence_threshold is None:
        confidence_threshold = suggestions[selection_mode]['confidence']
    
    if max_edges is None:
        max_edges = suggestions[selection_mode]['max_edges']
    
    
    final_dag = nx.DiGraph()
    for col in column_names:
        final_dag.add_node(col)
    
    
    sorted_edges = sorted(edges_info.items(), key=lambda x: x[1]['importance'], reverse=True)
    
    added_edges = 0
    for (u, v), info in sorted_edges:
        
        if info['confidence'] < confidence_threshold:
            continue
        
       
        if added_edges >= max_edges:
            break
        
        
        test_dag = final_dag.copy()
        test_dag.add_edge(u, v)
        if not list(nx.simple_cycles(test_dag)):
            
            final_dag.add_edge(u, v, **info)
            added_edges += 1
    
    return edges_info, final_dag, suggestions


def visualize_filtered_dag(
    dag: nx.DiGraph, 
    target_var: str,
    suggestions: Dict,
    output_file: str = 'filtered_dag'
):
    
    plt.figure(figsize=(12, 8))
    
    
    pos = nx.spring_layout(dag, k=2, iterations=50)
    
    
    node_colors = ['lightpink' if node == target_var else 'lightblue' for node in dag.nodes()]
    nx.draw_networkx_nodes(
        dag, pos,
        node_color=node_colors,
        node_size=2000,
        alpha=0.7
    )
   
    nx.draw_networkx_labels(dag, pos, font_size=10, font_weight='bold')
    
   
    edges = dag.edges(data=True)
    edge_colors = ['blue' for _ in edges]
    edge_widths = [1 + 2 * d['importance'] for _, _, d in edges]  
    
    nx.draw_networkx_edges(
        dag, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=20
    )
    
    
    edge_labels = {(u, v): f"{d['importance']:.3f}" for u, v, d in dag.edges(data=True)}
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title('Directed Acyclic Graph (DAG) Visualization', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(8, 6))
    importance_scores = [d.get('importance', 0) for _, _, d in dag.edges(data=True)]
    sns.histplot(importance_scores, bins=20, kde=True, color='blue')
    plt.title('Distribution of Edge Importance Scores')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_file}_distribution.png', dpi=300)
    plt.close()


def main():
    
    try:
        
        data = pd.read_csv('combined_data.csv')
        
        
        print("Columns in dataset:", data.columns.tolist())
        
        
        target_var = 'PP'       
        n_bootstraps = 100     
        alpha = 0.4      
        confidence_threshold = 0.0001
        max_edges = 20
        
        print(f"Using '{target_var}' as target variable.")
        print(f"Number of bootstrap iterations: {n_bootstraps}")
        print(f"Alpha level: {alpha}")
        print(f"Confidence threshold: {confidence_threshold}")
        
        
        edges_info, final_dag, suggestions = bootstrap_pc_algorithm(
            data=data,  
            target_var=target_var,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            confidence_threshold=confidence_threshold,
            max_edges=max_edges,
            selection_mode='liberal'  
        )
        
        
        print("\nAnalysis complete!")
        print(f"Number of nodes: {final_dag.number_of_nodes()}")
        print(f"Number of edges: {final_dag.number_of_edges()}")
        
       
        visualize_filtered_dag(final_dag, target_var, suggestions, output_file='filtered_dag')
        print("Visualizations saved: 'filtered_dag.png' & 'filtered_dag_distribution.png'")
        
        
        print("\nEdges information:")
        for edge, info in edges_info.items():
            print(f"{edge}: {info}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

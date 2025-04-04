import numpy as np
import pandas as pd
import networkx as nx
import ges
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(data):
    
   
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 添加小量噪声以减少共线性
    noise = np.random.normal(0, 1e-6, scaled_data.shape)
    processed_data = scaled_data + noise
    
    return processed_data

def remove_cycles(G, target_idx, variable_names=None, debug=True):
    
    while True:
        try:
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
                
            if debug:
                print("\nRemoving cycles:")
            
            for cycle in cycles:
                if debug:
                    if variable_names:
                        cycle_names = [variable_names[i] for i in cycle]
                        print(f"Cycle found: {cycle_names}")
                    else:
                        print(f"Cycle found: {cycle}")
                
                
                edges_in_cycle = [(cycle[i], cycle[(i+1)%len(cycle)]) 
                                for i in range(len(cycle))]
                
                
                removed = False
                for edge in edges_in_cycle:
                    if (edge[1] != target_idx and  
                        G.has_edge(edge[0], edge[1])):  
                        G.remove_edge(*edge)
                        if debug:
                            if variable_names:
                                print(f"Removed edge: {variable_names[edge[0]]} -> {variable_names[edge[1]]}")
                            else:
                                print(f"Removed edge: {edge}")
                        removed = True
                        break
                
               
                if not removed and edges_in_cycle:
                    for edge in edges_in_cycle:
                        if G.has_edge(edge[0], edge[1]):
                            G.remove_edge(*edge)
                            if debug:
                                if variable_names:
                                    print(f"Forced to remove edge: {variable_names[edge[0]]} -> {variable_names[edge[1]]}")
                                else:
                                    print(f"Forced to remove edge: {edge}")
                            break
        
        except nx.NetworkXNoCycle:
            break
    
    return G

def run_single_ges(data, variable_names, target_var):
    
    target_idx = variable_names.index(target_var)
    n_vars = len(variable_names)
    
    try:
       
        processed_data = preprocess_data(data)
        
        
        A, score = ges.fit_bic(processed_data)
        
    except np.linalg.LinAlgError:
       
        print("Warning: GES algorithm failed, using correlation-based approach")
        
        
        corr_matrix = np.abs(np.corrcoef(data.T))
        
       
        A = np.zeros((n_vars, n_vars))
        threshold = 0.1  
        
       
        for i in range(n_vars):
            if i != target_idx:
                if corr_matrix[i, target_idx] > threshold:
                    A[i, target_idx] = 1
    
   
    G = nx.DiGraph(A)
    
    
    G = remove_cycles(G, target_idx, variable_names, debug=False)
    
    
    for i in range(n_vars):
        if i != target_idx and G.out_degree(i) == 0:
            G.add_edge(i, target_idx)
    
    return nx.to_numpy_array(G, nodelist=range(n_vars))

def bootstrap_dag(data, variable_names, target_var, n_bootstrap=100, threshold=0.5):
    
    n_samples, n_vars = data.shape
    target_idx = variable_names.index(target_var)
    
    all_dags = np.zeros((n_bootstrap, n_vars, n_vars))
    failed_iterations = 0
    
    print("\nRunning bootstrap iterations:")
    for i in range(n_bootstrap):
        try:
           
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = data[indices]
            
           
            all_dags[i] = run_single_ges(bootstrap_data, variable_names, target_var)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} iterations")
                
        except Exception as e:
            print(f"Warning: Iteration {i+1} failed: {str(e)}")
            failed_iterations += 1
            continue
    
    if failed_iterations > 0:
        print(f"\nWarning: {failed_iterations} iterations failed and were skipped")
        
   
    if failed_iterations == n_bootstrap:
        print("\nWarning: All iterations failed, using correlation-based approach")
        corr_matrix = np.abs(np.corrcoef(data.T))
        final_dag = np.zeros((n_vars, n_vars))
        threshold = 0.1
        
        for i in range(n_vars):
            if i != target_idx and corr_matrix[i, target_idx] > threshold:
                final_dag[i, target_idx] = 1
                
        edge_frequencies = final_dag.copy()
        G = nx.DiGraph(final_dag)
        return final_dag, edge_frequencies, G
    
    
    successful_iterations = n_bootstrap - failed_iterations
    edge_frequencies = np.sum(all_dags, axis=0) / successful_iterations
    
    
    final_dag = np.zeros((n_vars, n_vars))
    final_dag[edge_frequencies >= threshold] = 1
    
   
    for i in range(n_vars):
        if i != target_idx and np.sum(final_dag[i, :]) == 0:
            final_dag[i, target_idx] = 1
            edge_frequencies[i, target_idx] = threshold
    
   
    G = nx.DiGraph(final_dag)
    G = remove_cycles(G, target_idx, variable_names)
    final_dag = nx.to_numpy_array(G, nodelist=range(n_vars))
    
    return final_dag, edge_frequencies, G

def plot_dag_matplotlib(G, variable_names, target_var, edge_frequencies=None, filename='causal_dag'):
   
    plt.figure(figsize=(12, 8))
    
   
    try:
       
        pos = nx.spring_layout(G, k=0.5, iterations=100)
    except np.linalg.LinAlgError:
        try:
           
            pos = nx.kamada_kawai_layout(G)
        except:
            try:
               
                pos = nx.circular_layout(G)
            except:
               
                pos = nx.shell_layout(G)
    
    target_idx = variable_names.index(target_var)
    
   
    node_colors = ['lightpink' if i == target_idx else 'lightblue' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.7)
    
   
    nx.draw_networkx_labels(G, pos, 
                          {i: name for i, name in enumerate(variable_names)},
                          font_size=8)  
    
    
    if edge_frequencies is not None:
       
        edges = G.edges()
        edge_colors = []
        edge_widths = []
        
        for (u, v) in edges:
            freq = edge_frequencies[u, v]
            if v == target_idx:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
            edge_widths.append(1 + 2 * freq)
        
      
        nx.draw_networkx_edges(G, pos, 
                             edge_color=edge_colors, 
                             width=edge_widths,
                             arrowsize=15,  
                             arrowstyle='->',
                             connectionstyle='arc3,rad=0.1')  
    else:
        nx.draw_networkx_edges(G, pos, 
                             edge_color='black', 
                             width=1,
                             arrowsize=15,
                             arrowstyle='->',
                             connectionstyle='arc3,rad=0.1')
    
    plt.title('Causal DAG', pad=20)
    plt.axis('off')
    
   
    plt.tight_layout()
    
   
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

def get_edge_statistics(G, variable_names, target_var, edge_frequencies):
    
    target_idx = variable_names.index(target_var)
    
   
    direct_causes = list(G.predecessors(target_idx))
    
   
    indirect_causes = set()
    for node in direct_causes:
        ancestors = nx.ancestors(G, node)
        indirect_causes.update(ancestors)
    
   
    indirect_causes = indirect_causes - set(direct_causes)
    
   
    direct_causes = sorted(direct_causes, 
                         key=lambda x: edge_frequencies[x, target_idx],
                         reverse=True)
    
    return direct_causes, list(indirect_causes), target_idx

def main():
   
    try:
        data = pd.read_csv('combined_data.csv')
    except FileNotFoundError:
        print("Error: Please ensure 'combined_data.csv' exists in the current directory.")
        return

    # 准备数据
    variable_names = data.columns.tolist()
    target_var = 'PP'
    data_array = data.values
    
    # 运行bootstrap GES算法
    print(f"Starting causal structure learning with bootstrap (target: {target_var})")
    final_dag, edge_frequencies, G = bootstrap_dag(data_array, variable_names, target_var, 
                                                 n_bootstrap=1000, threshold=0.3)
    
    # 输出邻接矩阵
    print("\nFinal DAG adjacency matrix:")
    adj_matrix_df = pd.DataFrame(final_dag, 
                                index=variable_names, 
                                columns=variable_names)
    print(adj_matrix_df)
    
    # 输出频率矩阵
    print("\nEdge frequencies matrix:")
    freq_matrix_df = pd.DataFrame(edge_frequencies,
                                 index=variable_names,
                                 columns=variable_names)
    print(freq_matrix_df)
    
    # 检查是否有环
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print("\nWarning: Graph contains cycles:", cycles)
        else:
            print("\nGraph is acyclic")
    except nx.NetworkXNoCycle:
        print("\nGraph is acyclic")
    
    # 获取并输出因果关系统计
    direct_causes, indirect_causes, target_idx = get_edge_statistics(G, variable_names, target_var, edge_frequencies)
    
    print(f"\nVariables directly affecting {target_var} (sorted by frequency):")
    for idx in direct_causes:
        print(f"- {variable_names[idx]} (frequency: {edge_frequencies[idx, target_idx]:.3f})")
        
    print(f"\nVariables indirectly affecting {target_var}:")
    for idx in indirect_causes:
        print(f"- {variable_names[idx]}")
    
    # 绘制和保存DAG图
    print("\nGenerating causal DAG visualization...")
    plot_dag_matplotlib(G, variable_names, target_var, edge_frequencies)
    print("DAG visualization has been saved as 'causal_dag.pdf' and 'causal_dag.png'")

if __name__ == "__main__":
    main()

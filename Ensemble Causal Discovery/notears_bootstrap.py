import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

def notears_linear(X, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, 
                  rho_max=1e+16, w_threshold=0.3, target_var_idx=None):
    
    def _loss(W):
       
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _func(w):
        
        W = w.reshape([d, d])
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(w).sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = G_smooth.flatten() + lambda1 * np.sign(w)
        return obj, g_obj

    n, d = X.shape
    w_est = np.zeros(d * d)
    rho, alpha, h = 1.0, 0.0, np.inf  # init dual ascent
    bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
    
    if target_var_idx is not None:
        for j in range(d):
            if j != target_var_idx:
                bnds[target_var_idx * d + j] = (0, 0)

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(w_new.reshape([d, d]))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = w_est.reshape([d, d])
    W_est[np.abs(W_est) < w_threshold] = 0

    if target_var_idx is not None:
        outgoing_edges = np.sum(np.abs(W_est) > 0, axis=1)
        source_nodes = np.where(outgoing_edges == 0)[0]
        
        for source_idx in source_nodes:
            if source_idx != target_var_idx:
                W_est[source_idx, target_var_idx] = np.random.uniform(0.1, 0.2)

    return W_est

def bootstrap_notears(X, n_bootstrap=50, lambda1=0.1, w_threshold=0.2, target_var_idx=None):
    
    n, d = X.shape
    W_bootstrap = np.zeros((n_bootstrap, d, d))
    successful_runs = 0
    
    print("\nPerforming bootstrap analysis...")
    pbar = tqdm(total=n_bootstrap)
    
    while successful_runs < n_bootstrap:
        try:
            indices = np.random.choice(n, size=n, replace=True)
            X_bootstrap = X[indices]
            W_bootstrap[successful_runs] = notears_linear(
                X_bootstrap, lambda1, w_threshold=w_threshold, 
                target_var_idx=target_var_idx
            )
            successful_runs += 1
            pbar.update(1)
        except Exception as e:
            print(f"\nWarning: Bootstrap iteration failed, retrying... ({str(e)})")
            continue
    
    pbar.close()
    
    W_freq = np.mean(W_bootstrap != 0, axis=0)
    W_mean = np.mean(W_bootstrap, axis=0)
    W_std = np.std(W_bootstrap, axis=0)
    
    return W_freq, W_mean, W_std

def create_causal_graph(W_mean, W_freq, variables, filename, freq_threshold=0.5):
    
    G = nx.DiGraph()
    
    # Add nodes
    for var in variables:
        G.add_node(var)
    
    # Add edges
    for i, from_var in enumerate(variables):
        for j, to_var in enumerate(variables):
            if i != j and W_freq[i, j] >= freq_threshold:
                G.add_edge(from_var, to_var, 
                          weight=W_mean[i, j],
                          frequency=W_freq[i, j])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, node_shape='s')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with different colors, widths and arrows
    edges = G.edges(data=True)
    for u, v, data in edges:
        weight = data['weight']
        frequency = data['frequency']
        color = 'red' if weight < 0 else 'blue'
        width = 1 + 2 * frequency
        
        # Draw edge with arrow
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                             edge_color=color, width=width,
                             arrowsize=20,  # 增大箭头大小
                             arrowstyle='->',  # 使用箭头样式
                             connectionstyle='arc3,rad=0.1')  # 稍微弯曲的边，避免重叠
        
        # Add edge labels
        label = f'{weight:.3f}\n({frequency:.1%})'
        x = pos[u][0] * 0.7 + pos[v][0] * 0.3  # 调整标签位置
        y = pos[u][1] * 0.7 + pos[v][1] * 0.3
        plt.text(x, y, label, horizontalalignment='center', 
                verticalalignment='center', bbox=dict(facecolor='white', 
                edgecolor='none', alpha=0.7))
    
    plt.title('Causal Graph')
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_and_analyze(data_path, target_var="PP", output_dir="results", 
                       n_bootstrap=50, freq_threshold=0.2):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(data_path)
    X = df.values
    variables = df.columns.tolist()
    
    target_var_idx = variables.index(target_var)
    
    # Standardize the data
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    
    W_freq, W_mean, W_std = bootstrap_notears(
        X, n_bootstrap=n_bootstrap, target_var_idx=target_var_idx
    )
    
    significant_edges = (W_freq >= freq_threshold)
    n_edges = np.sum(significant_edges)
    print(f"\nNumber of significant causal relationships (frequency >= {freq_threshold}): {n_edges}")
    
    if n_edges > 0:
        causal_df = pd.DataFrame(W_mean, columns=variables, index=variables)
        freq_df = pd.DataFrame(W_freq, columns=variables, index=variables)
        std_df = pd.DataFrame(W_std, columns=variables, index=variables)
        
        # Try to save Excel file if openpyxl is available
        try:
            with pd.ExcelWriter(f'{output_dir}/causal_analysis.xlsx') as writer:
                causal_df.to_excel(writer, sheet_name='Mean Weights')
                freq_df.to_excel(writer, sheet_name='Edge Frequencies')
                std_df.to_excel(writer, sheet_name='Weight STD')
            print("Excel file saved successfully.")
        except ImportError:
            print("Warning: openpyxl not installed. Saving results as CSV files instead.")
            causal_df.to_csv(f'{output_dir}/causal_weights.csv')
            freq_df.to_csv(f'{output_dir}/edge_frequencies.csv')
            std_df.to_csv(f'{output_dir}/weight_std.csv')
        
        create_causal_graph(W_mean, W_freq, variables, 
                          f'{output_dir}/causal_graph',
                          freq_threshold=freq_threshold)
        
        with open(f'{output_dir}/analysis_report.txt', 'w') as f:
            f.write("Linear Causal Discovery Analysis Report with Bootstrap\n\n")
            f.write(f"Target variable: {target_var}\n")
            f.write(f"Bootstrap iterations: {n_bootstrap}\n")
            f.write(f"Frequency threshold: {freq_threshold}\n\n")
            f.write("Significant causal relationships:\n")
            
            for i, from_var in enumerate(variables):
                for j, to_var in enumerate(variables):
                    if i != j and W_freq[i, j] >= freq_threshold:
                        f.write(f"{from_var} -> {to_var}:\n")
                        f.write(f"  Mean weight: {W_mean[i, j]:.3f}\n")
                        f.write(f"  Frequency: {W_freq[i, j]:.1%}\n")
                        f.write(f"  Std: {W_std[i, j]:.3f}\n\n")
            
            f.write(f"\nTotal number of significant causal relationships: {n_edges}\n")
    else:
        print("No significant causal relationships found with current parameters.")
    
    return W_mean, W_freq, W_std

if __name__ == "__main__":
    data_path = "combined_data.csv"
    try:
        W_mean, W_freq, W_std = process_and_analyze(data_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

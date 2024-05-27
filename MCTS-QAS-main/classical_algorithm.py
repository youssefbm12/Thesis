
'''
import numpy as np
import networkx as nx
from scipy.linalg import eigh

def goemans_williamson_max_cut(graph):
    # Create the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(graph)

    # Compute the Laplacian matrix
    n = len(graph.nodes)
    D = np.diag(np.array(adj_matrix.sum(axis=1)).flatten())
    L = D - adj_matrix

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(L, eigvals=(n-2, n-1))

    # Extract the eigenvector corresponding to the second smallest eigenvalue
    min_eigvec = eigenvectors[:, 0]

    # Assign vertices to sets based on the sign of the eigenvector components
    binary_result = np.array([1 if min_eigvec[i] >= 0 else 0 for i in range(n)])

    return binary_result

# Example usage:
# Define the graph
edges = [(0, 1), (0, 3), (0, 5), (2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5), (6, 1), (6, 3), (6, 5)]
G = nx.Graph()
G.add_edges_from(edges)

# Find the Max-Cut using Goemans-Williamson algorithm
binary_result = goemans_williamson_max_cut(G)
print("Binary Result:", binary_result)





def brute_force_maxcut(graph):
    nodes = list(graph.nodes())
    n = len(nodes)
    max_cut_size = 0
    best_partitions = []

    # Iterate over all possible subsets of nodes, except the empty set and the full set
    for i in range(1, 2 ** n // 2):
        # Create a subset of nodes
        subset = [nodes[j] for j in range(n) if i & (1 << j)]
        complement = list(set(nodes) - set(subset))
        
        # Calculate the number of edges between the subset and its complement
        cut_size = sum(1 for u, v in graph.edges() if (u in subset and v in complement) or (u in complement and v in subset))
        
        # Check if this cut is better than what we've found so far
        if cut_size > max_cut_size:
            max_cut_size = cut_size
            best_partitions = [(subset, complement)]
        elif cut_size == max_cut_size:
            best_partitions.append((subset, complement))

    # Convert partitions to binary string representation
    binary_partitions = []
    for partition in best_partitions:
        binary_string = ''.join(['1' if node in partition[0] else '0' for node in nodes])
        binary_partitions.append(binary_string)

    return max_cut_size, binary_partitions

'''
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def goemans_williamson_maxcut_mock(G):
    n = len(G.nodes)
    
    # Mock solution to the SDP (for demonstration purposes)
    X_opt = np.eye(n)
    
    # Random hyperplane rounding
    D = np.linalg.cholesky(X_opt)
    random_vector = np.random.randn(n)
    cut = np.sign(D @ random_vector)
    
    # Partition the nodes based on the cut
    partition_1 = [i for i in range(n) if cut[i] >= 0]
    partition_2 = [i for i in range(n) if cut[i] < 0]
    
    return partition_1, partition_2

def plot_maxcut(G, partition_1, partition_2, num_edges_cut, cut_value):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=partition_1, node_color='r', label='Partition 1')
    nx.draw_networkx_nodes(G, pos, nodelist=partition_2, node_color='b', label='Partition 2')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    plt.legend()
    plt.title(f'Goemans-Williamson Max-Cut (Mock)\nNumber of edges cut: {num_edges_cut}, Cut value: {cut_value}')
    plt.show()

def count_edges_cut(G, partition_1, partition_2):
    cut_edges = 0
    cut_value = 0
    for u, v in G.edges():
        if (u in partition_1 and v in partition_2) or (u in partition_2 and v in partition_1):
            cut_edges += 1
            cut_value += 1  # Assuming each edge has weight 1
    return cut_edges, cut_value

# Create the graph from the given edges
edges = [(0, 1), (0, 3), (0, 5), (2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5), (6, 1), (6, 3), (6, 5)]
G = nx.Graph()
G.add_edges_from(edges)

# Apply the Goemans-Williamson algorithm (mocked)
partition_1, partition_2 = goemans_williamson_maxcut_mock(G)

# Count the number of edges cut and calculate the cut value
num_edges_cut, cut_value = count_edges_cut(G, partition_1, partition_2)
print(f"Number of edges cut: {num_edges_cut}")
print(f"Value of the cut: {cut_value}")
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import cvxpy as cp
from problems.combinatorial import graph
def goemans_williamson_maxcut(G, epsilon=1e-10):
    n = len(G.nodes)
    W = nx.adjacency_matrix(G).todense()

    # SDP variables
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]
    objective = cp.Maximize(0.25 * cp.trace(W @ (np.ones((n, n)) - X)))

    # Solve the SDP
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Retrieve the solution
    X_opt = X.value

    # Ensure the matrix is positive definite
    X_opt += epsilon * np.eye(n)

    try:
        # Cholesky decomposition
        D = np.linalg.cholesky(X_opt)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(X_opt)
        eigvals[eigvals < 0] = 0  # Ensure non-negative eigenvalues
        D = eigvecs @ np.diag(np.sqrt(eigvals))

    random_vector = np.random.randn(n)
    cut = np.sign(D @ random_vector)

    # Partition the nodes based on the cut
    partition_1 = [i for i in range(n) if cut[i] >= 0]
    partition_2 = [i for i in range(n) if cut[i] < 0]
    partition = ''.join(['1' if i in partition_1 else '0' for i in range(n)])

    return partition

def brute_force_maxcut(G):
    n = len(G.nodes)
    max_cut_value = 0
    best_partition = '0' * n
    
    # Iterate over all possible partitions
    for i in range(1, n//2 + 1):
        for nodes in itertools.combinations(G.nodes, i):
            partition_1 = list(nodes)
            partition_2 = [node for node in G.nodes if node not in partition_1]
            
            cut_value = count_edges_cut(G, partition_1, partition_2)[1]
            
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                best_partition = ''.join(['1' if node in partition_1 else '0' for node in range(n)])
    
    return best_partition, max_cut_value

def count_edges_cut(G, partition_1, partition_2):
    cut_edges = 0
    cut_value = 0
    for u, v in G.edges():
        if (u in partition_1 and v in partition_2) or (u in partition_2 and v in partition_1):
            cut_edges += 1
            cut_value += 1  # Assuming each edge has weight 1
    return cut_edges, cut_value

def partition_from_string(partition_str):
    partition_1 = [i for i, bit in enumerate(partition_str) if bit == '1']
    partition_2 = [i for i, bit in enumerate(partition_str) if bit == '0']
    return partition_1, partition_2

def plot_maxcut_comparison(G, gw_partition_str, bf_partition_str, gw_cut_value, bf_cut_value):
    pos = nx.spring_layout(G)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    gw_partition_1, gw_partition_2 = partition_from_string(gw_partition_str)
    bf_partition_1, bf_partition_2 = partition_from_string(bf_partition_str)
    
    # Plot Goemans-Williamson result
    nx.draw(G, pos, ax=axes[0], with_labels=True, node_color=['r' if node in gw_partition_1 else 'b' for node in G.nodes])
    axes[0].set_title(f'Goemans-Williamson Max-Cut\nCut value: {gw_cut_value}')
    
    # Plot brute-force result
    nx.draw(G, pos, ax=axes[1], with_labels=True, node_color=['r' if node in bf_partition_1 else 'b' for node in G.nodes])
    axes[1].set_title(f'Brute Force Max-Cut\nCut value: {bf_cut_value}')
    
    plt.show()

# Create the graph from the given edges
edges = graph
G = nx.Graph()
G.add_edges_from(edges)

# Apply the Goemans-Williamson algorithm
gw_partition_str = goemans_williamson_maxcut(G)
gw_partition_1, gw_partition_2 = partition_from_string(gw_partition_str)
gw_cut_edges, gw_cut_value = count_edges_cut(G, gw_partition_1, gw_partition_2)
print(f"Goemans-Williamson - Partition: {gw_partition_str}, Number of edges cut: {gw_cut_edges}, Cut value: {gw_cut_value}")

# Apply the brute force method
bf_partition_str, bf_cut_value = brute_force_maxcut(G)
bf_partition_1, bf_partition_2 = partition_from_string(bf_partition_str)
bf_cut_edges, bf_cut_value = count_edges_cut(G, bf_partition_1, bf_partition_2)
print(f"Brute Force - Partition: {bf_partition_str}, Number of edges cut: {bf_cut_edges}, Cut value: {bf_cut_value}")

# Plot the results side by side
#plot_maxcut_comparison(G, gw_partition_str, bf_partition_str, gw_cut_value, bf_cut_value)
solution = bf_partition_str

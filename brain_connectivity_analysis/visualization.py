import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import networkx as nx

# Load variables from data_preprocessed.pickle
with open('manage_data/data_preprocessed.pickle', 'rb') as f:
    connectivity_matrices, controls, patients, controls_count, patients_count, subject_count, patient_info_dict, responders, non_responders, response_df, medication = pickle.load(f)

# Basic network manipulation functions

def plot_connectivity_matrix(*matrix, subject_name=None, save=None):
    """
        Plot the connectivity matrix as a colormap
        It uses a logarithmic scale for the colors and it is possible to plot 
        multiple matrices at once.
    """
    
    max_rows = 2
    line = int(len(matrix) / 3 + 1)
    
    for i, mat in enumerate(matrix):
        plt.subplot(line, max_rows, i+1)
        plt.grid(False)
        plt.imshow(np.log(mat+1),cmap='viridis')
        plt.xticks(np.arange(0, 81, 10))
        plt.yticks(np.arange(0, 81, 10))
        plt.xlabel('ROIs')
        plt.ylabel('ROIs')
        plt.title(subject_name)
    
    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save,bbox_inches='tight')
        
def get_network(matrix, threshold = 0):
    """ 
        Return the network (as a networkx data structure) defined by matrix.
        It is possible to specify a threshold that will disregard all the 
        edges below this threshold when creating the network
    """
    G = nx.Graph()
    N = matrix.shape[0]
    G.add_nodes_from(list(range(N)))
    G.add_weighted_edges_from([(i,j,1.0*matrix[i][j]) for i in range(0,N) for j in range(0,i) \
                                                                   if matrix[i][j] >= threshold])
    return G

print("Some connectivity matrices: {} and {}".format(controls[0], patients[0]))
plot_connectivity_matrix(connectivity_matrices[controls[0]], connectivity_matrices[patients[0]])

print("Compute the network associated to a subject ({})".format(controls[0]))
G = get_network(connectivity_matrices[controls[0]])
nx.draw(G)
plt.show()

#%% compare two sets of connectivity matrices 
for patient in connectivity_matrices.keys():
    plot_connectivity_matrix(tmp[patient], connectivity_matrices[patient], subject_name=patient)
#%% Graph metrics 

def getGraphMetrics(graph):
    
    graph_degree = dict(graph.degree)
    print("Graph Summary:")
    print(f"Number of nodes : {len(graph.nodes)}")
    print(f"Number of edges : {len(graph.edges)}")
    print(f"Maximum degree : {np.max(list(graph_degree.values()))}")
    print(f"Minimum degree : {np.min(list(graph_degree.values()))}")
    print(f"Average degree : {np.mean(list(graph_degree.values()))}")
    print(f"Median degree : {np.median(list(graph_degree.values()))}")
    print("")
    print("Graph Connectivity")
    try:
        print(f"Connected Components : {nx.number_connected_components(graph)}")
    except:
        print(f"Strongly Connected Components : {nx.number_strongly_connected_components(graph)}")
        print(f"Weakly Connected Components : {nx.number_weakly_connected_components(graph)}")
    print("")
    print("Graph Distance")
    print(f"Average Distance : {nx.average_shortest_path_length(graph)}")
    print(f"Diameter : {nx.algorithms.distance_measures.diameter(graph)}")
    print("")
    print("Graph Clustering")
    print(f"Transitivity : {nx.transitivity(graph)}")
    print(f"Average Clustering Coefficient : {nx.average_clustering(graph)}")
    
    
    return None

G = connectivity_matrices[patients[0]]
G_nx = get_network(G)
getGraphMetrics(G_nx)

degree_freq = np.array(nx.degree_histogram(G_nx)).astype('float')

plt.rcParams.update({'font.size': 12})

ax1 = sns.displot(degree_freq, kde=True)
ax1.set(ylabel="Frequency", xlabel="Degree", title="Distribution of degree")
plt.show()

betweenness_centrality = pd.DataFrame(nx.betweenness_centrality(G_nx).items(), columns=['Region', 'Betweenness centrality'])
print(betweenness_centrality.head())

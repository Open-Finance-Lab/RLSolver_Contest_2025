import os
import pickle
import networkx as nx
import re

def _nat_key(fname: str):
    """
    Extracts integer key from filename for natural sorting.
    Example: DRL_4_ID2 < DRL_4_ID10 < DRL_4_ID11.
    
    Tries to match _ID\d+ pattern first, otherwise uses first digit sequence.
    """
    m = re.search(r'_ID(\d+)', fname)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else fname


def txt_folder_to_eco_pkl(input_folder: str, output_pkl: str):
    """
    Reads all .txt files in `input_folder` (GSET-like format) and saves a .pkl
    file containing a list of NetworkX Graphs, compatible with ECO-DQN's
    load_graph_set utility.
    
    Assumed file format:
      - First line: <num_nodes> <num_edges>
      - Subsequent lines: i j weight   (1-based node indices, float weights)
    """
    graphs = []
    for fname in sorted(os.listdir(input_folder), key=_nat_key):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(input_folder, fname)
        with open(path, 'r') as f:
            header = f.readline().split()
            if len(header) < 2:
                raise ValueError(f"Invalid header in {fname}")
            n_nodes = int(header[0])
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))  # nodes 0..n_nodes-1
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                i, j, w = parts
                u = int(i) - 1  # convert 1-based → 0-based indexing
                v = int(j) - 1
                G.add_edge(u, v, weight=float(w))
        graphs.append(G)
        print(f"Loaded {fname} → {n_nodes} nodes, {G.number_of_edges()} edges")
    
    # Serialize the list of graphs
    with open(output_pkl, 'wb') as out:
        pickle.dump(graphs, out)
    print(f"Saved {len(graphs)} graphs to {output_pkl}")

# Example usage:
if __name__ == "__main__":
    # Convert all .txt files in 'data' folder to a single eco-compatible .pkl file
    # Adjust the input folder and output file name as needed
    txt_folder_to_eco_pkl('data', 'eco_graphs.pkl')

import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edge(1, 2, weight=4.7)

try:
    nx.write_gpickle(G, "test_graph.gpickle")
    print("write_gpickle function works correctly.")
except AttributeError as e:
    print(f"AttributeError: {e}")

import pydot

# Create a new graph
graph = pydot.Dot(graph_type='digraph', rankdir='LR')
graph.set_ranksep('3.0')

# Function to create a subgraph for a layer
def create_layer_input(name, num_nodes, label):
    subgraph = pydot.Cluster(name, label=label, style='filled', color='white', fontsize='20', fontname='Helvetica-bold')
    nodes = []
    names = ["Incidence\nLags", 
             "Large City\nIncidence\nLags", 
             "Large City \nDistances",
             "Near City\nIncidence\nLags", 
             "Near City\nDistances",
             "Population",
             "Births"]
    for i in range(num_nodes):
        node = pydot.Node(names[i], shape='circle', width='1.5', height='1.5', fontsize='16', fontname='Helvetica-bold')
        subgraph.add_node(node)
        nodes.append(node)
    graph.add_subgraph(subgraph)
    return nodes

def create_layer(name, num_nodes, label):
    subgraph = pydot.Cluster(name, label=label, style='filled', color='white', fontsize='20', fontname='Helvetica-bold')
    nodes = []
    for i in range(num_nodes):
        node = pydot.Node(f'{name}_{i}', label='', shape='circle', width='0.9', height='0.9', fontsize='16', fontname='Helvetica-bold')
        subgraph.add_node(node)
        nodes.append(node)
    graph.add_subgraph(subgraph)
    return nodes

def create_layer_output(name, num_nodes, label):
    subgraph = pydot.Cluster(name, label=label, style='filled', color='white', fontsize='20', fontname='Helvetica-bold')
    nodes = []
    names = ["Incidence\nForecast"]
    for i in range(num_nodes):
        node = pydot.Node(names[i], shape='circle', width='1.5', height='1.0', fontsize='16', fontname='Helvetica-bold')
        subgraph.add_node(node)
        nodes.append(node)
    graph.add_subgraph(subgraph)
    return nodes

# Create layers and add edges as previously described


# Create layers
input_nodes = create_layer_input('input', 7, 'Input Layer')
hidden1_nodes = create_layer('hidden1', 6, 'Hidden Layer 1')
hidden2_nodes = create_layer('hidden2', 6, 'Hidden Layer 2')
hidden2_nodes = create_layer('hidden2', 6, 'Hidden Layer 3')
output_nodes = create_layer_output('output', 1, 'Output Layer')

# Function to add edges between layers
def add_edges(from_nodes, to_nodes):
    for f in from_nodes:
        for t in to_nodes:
            graph.add_edge(pydot.Edge(f, t))

# Add edges between layers
add_edges(input_nodes, hidden1_nodes)
add_edges(hidden1_nodes, hidden2_nodes)
add_edges(hidden2_nodes, output_nodes)

# Save the diagram to a file
graph.write_png('../../../../output/figures/basic_nn/feedforward_network_structure.png')


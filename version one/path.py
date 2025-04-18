import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Path:
    """Represents a path through nodes in a graph with associated cost"""
    def __init__(self, start_node):
        self.nodes = [start_node]  # Ordered list of nodes in the path
        self.cost = 0              # Total cost of the path
        
    def add_node(self, node, edge_cost):
        """Add a node to the end of the path
            node: The node to add
            edge_cost: Cost from the last node to this new node"""
        self.nodes.append(node)
        self.cost += edge_cost
        
    def contains(self, node):
        """Check if a node exists in the path
        Returns:
            bool: True if node is in path, False otherwise"""
        return node in self.nodes
    def cost_to_node(self, node):
        """Calculate total cost from start to the specified node
        Returns:float: Total cost to reach node, or -1 if node not in path"""
        total_cost = 0
        for i in range(len(self.nodes)):
            if self.nodes[i] == node:
                return total_cost
            if i < len(self.nodes) - 1:
                total_cost += self.nodes[i].get_edge_cost(self.nodes[i+1])
        return -1
        
    def clone(self):
        """ Create an identical copy of this path
        Returns:Path: A new Path object with identical nodes and cost"""
        new_path = Path(self.nodes[0])
        new_path.nodes = self.nodes.copy()
        new_path.cost = self.cost
        return new_path

def plot_path(graph, path):
    """Visualize a graph with a highlighted path"""
    plt.figure(figsize=(10, 8))
    
    # Plot all nodes
    all_nodes = graph.get_all_nodes()
    for node in all_nodes:
        plt.plot(node.x, node.y, 'o', color='lightgray', markersize=12)
        plt.text(node.x, node.y, node.name, 
                ha='center', va='center', fontsize=10)
    
    # Plot all edges
    for node in all_nodes:
        for neighbor in graph.get_neighbors(node):
            plt.plot([node.x, neighbor.x], [node.y, neighbor.y], 
                    'lightgray', linewidth=1)
    
    # Highlight path nodes
    if path and len(path.nodes) > 0:
        for node in path.nodes:
            plt.plot(node.x, node.y, 'o', color='red', markersize=15)
    
    # Highlight path edges
    if path and len(path.nodes) > 1:
        for i in range(len(path.nodes) - 1):
            x1, y1 = path.nodes[i].x, path.nodes[i].y
            x2, y2 = path.nodes[i+1].x, path.nodes[i+1].y
            plt.plot([x1, x2], [y1, y2], 'red', linewidth=3)
    
    # Add title with path cost if available
    if path:
        plt.title(f"Path Visualization (Total Cost: {path.cost:.2f})", pad=20)
    else:
        plt.title("Graph Visualization", pad=20)
    
    # Configure plot appearance
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
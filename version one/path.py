#VERSION 2 NEW FILE path.py
# path.py
import math
from matplotlib import pyplot as plt

from Graph import Plot
# path.py
import math
from matplotlib import pyplot as plt

# Remove the import from Graph and instead define a Plot function here
def Plot(graph, ax=None):
    """Basic plotting function for path.py to avoid circular imports"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw segments
    for segment in graph.segments.values():
        ax.plot([segment.origin.x, segment.destination.x],
                [segment.origin.y, segment.destination.y],
                'k-', alpha=0.5)
    
    # Draw nodes
    for node in graph.nodes.values():
        ax.plot(node.x, node.y, 'bo')
        ax.text(node.x, node.y, node.name, fontsize=12, ha='right')
    
    ax.grid(True)
    if ax is None:
        plt.show()

class Path:
    def __init__(self, start_node=None):
        self.nodes = []
        self.cost = 0
        if start_node:
            self.nodes.append(start_node)
    
    def AddNodeToPath(self, node, cost=1):
        """Add a node to the path with an associated cost"""
        if self.nodes:
            self.cost += cost
        self.nodes.append(node)
    
    def ContainsNode(self, node):
        """Check if node is in the path"""
        return node in self.nodes
    
    def CostToNode(self, node):
        """Calculate total cost to reach a node in the path"""
        if node not in self.nodes:
            return -1
        
        total_cost = 0
        for i in range(len(self.nodes)-1):
            current = self.nodes[i]
            next_node = self.nodes[i+1]
            
            # Find the segment between current and next_node
            for seg in current.neighbors:
                if seg.destination == next_node:
                    total_cost += seg.cost
                    break
            
            if self.nodes[i] == node:
                break
        
        return total_cost
    
    def GetLastNode(self):
        """Get the last node in the path"""
        if self.nodes:
            return self.nodes[-1]
        return None
    
    def Copy(self):
        """Create a copy of the path"""
        new_path = Path()
        new_path.nodes = self.nodes.copy()
        new_path.cost = self.cost
        return new_path
    
    def __str__(self):
        return " -> ".join([node.name for node in self.nodes]) + f" (Cost: {self.cost:.2f})"

def PlotPath(graph, path, ax=None):
    """Visualize the path on the graph"""
    if not path or len(path.nodes) < 2:
        return
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Draw the entire graph first
    Plot(graph, ax)
    
    # Highlight the path
    for i in range(len(path.nodes)-1):
        start = path.nodes[i]
        end = path.nodes[i+1]
        ax.plot([start.x, end.x], [start.y, end.y], 'r-', linewidth=3)
        ax.plot(start.x, start.y, 'go', markersize=10)
        ax.plot(end.x, end.y, 'go', markersize=10)
    
    # Label the path
    ax.set_title(f"Path: {path}\nTotal Cost: {path.cost:.2f}")
    
    if ax is None:
        plt.show()
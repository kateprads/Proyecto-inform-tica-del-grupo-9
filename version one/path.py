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


def PlotPath(g, path, ax=None):
    """Highlights path WITHOUT moving/changing original graph"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. FIRST get the current axis limits to lock them
    xlim = ax.get_xlim() if ax.lines else None
    ylim = ax.get_ylim() if ax.lines else None
    
    # 2. Draw ORIGINAL GRAPH (if not already drawn)
    if not ax.lines:  # Only plot if empty
        Plot(g, ax)
    
    # 3. RESTORE original view if it existed
    if xlim and ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # 4. Highlight PATH (if exists)
    if path and len(path.nodes) > 1:
        # Store current view
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Draw path highlights
        for i in range(len(path.nodes)-1):
            seg = next((s for s in g.segments.values() 
                       if s.origin.name == path.nodes[i].name 
                       and s.destination.name == path.nodes[i+1].name), None)
            if seg:
                ax.annotate("",
                    xy=(seg.destination.x, seg.destination.y),
                    xytext=(seg.origin.x, seg.origin.y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color='#FFD700',  # Gold
                        linewidth=3,
                        alpha=1,
                        mutation_scale=20
                    ),
                    zorder=10  # Always on top
                )
        
        # Restore view after drawing
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
    
    ax.set_title(f"Path: {'→'.join(n.name for n in path.nodes)} | Cost: {path.cost:.2f}")
    plt.tight_layout()
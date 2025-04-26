import heapq
from matplotlib.path import Path
import matplotlib.pyplot as plt
import math


class Node:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.neighbors = []

class Segment:
    def __init__(self, name, origin, destination, cost=1):
        self.name = name
        self.origin = origin
        self.destination = destination
        self.cost = cost

class Graph:
    def __init__(self):
        self.nodes = {}  # Cambiado a diccionario
        self.segments = {}  # Cambiado a diccionario

    def find_node(self, name):
        return self.nodes.get(name)

def AddNode(g, n):
    if n.name in g.nodes:
        return False
    g.nodes[n.name] = n
    return True


def AddSegment(g, name, nameOrigin, nameDestination, cost=None):  # ← cost es opcional
    origin = g.find_node(nameOrigin)
    destination = g.find_node(nameDestination)

    if origin is None or destination is None:
        return False

    # Usar costo manual si existe, si no, calcular euclidiano
    if cost is None:
        dx = origin.x - destination.x
        dy = origin.y - destination.y
        cost = round(math.sqrt(dx**2 + dy**2), 3)

    segment = Segment(name, origin, destination, cost)
    g.segments[name] = segment
    origin.neighbors.append(destination)
    return True

def GetClosest(g, x, y):
    closest_node = None
    min_distance = float('inf')

    for node in g.nodes.values():
        distance = math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node


def Plot(g, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw segments with perfect bright blue arrows
    for segment in g.segments.values():
        ax.annotate("",
                    xy=(segment.destination.x, segment.destination.y),
                    xytext=(segment.origin.x, segment.origin.y),
                    arrowprops=dict(
                        arrowstyle="-|>",  # Filled arrow style
                        color='#00ccff',   # Your bright blue
                        linewidth=2.0,
                        alpha=0.9,
                        mutation_scale=15,  # Medium arrowhead
                        shrinkA=0,
                        shrinkB=0
                    ))
        
        # Cost label without background
        mid_x = (segment.origin.x + segment.destination.x) / 2
        mid_y = (segment.origin.y + segment.destination.y) / 2
        ax.text(mid_x, mid_y, f"{segment.cost:.1f}", 
                color='red', fontsize=10, ha='center', va='center')

    # Draw nodes with clean labels
    for node in g.nodes.values():
        ax.plot(node.x, node.y, 'bo', markersize=8)
        ax.text(node.x + 0.15, node.y + 0.15, node.name,  # Slight offset
                fontsize=12, ha='left', va='bottom', color='black')

    ax.grid(True)
    if ax is None:
        plt.show()

def PlotNode(g, node_name, ax=None):
    node = g.find_node(node_name)
    if node is None:
        return False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw base graph
    Plot(g, ax)
    
    # Highlight specific node
    ax.plot(node.x, node.y, 'ro', markersize=10)
    ax.text(node.x + 0.15, node.y + 0.15, node.name, 
            fontsize=12, ha='left', va='bottom', color='red')
    
    # Highlight connections
    for segment in g.segments.values():
        if segment.origin == node or segment.destination == node:
            ax.annotate("",
                        xy=(segment.destination.x, segment.destination.y),
                        xytext=(segment.origin.x, segment.origin.y),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color='red',
                            linewidth=2.5,
                            alpha=1.0,
                            mutation_scale=20
                        ))
    
    if ax is None:
        plt.show()
    return True 

def ReadGraphFromFile(filename):
    g = Graph()
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if parts[0] == 'NODE' and len(parts) == 4:
                    name, x, y = parts[1], float(parts[2]), float(parts[3])
                    AddNode(g, Node(name, x, y))
                elif parts[0] == 'SEGMENT' and len(parts) >= 4:
                    seg_name, origin, dest = parts[1], parts[2], parts[3]
                    cost = float(parts[4]) if len(parts) > 4 else 1.0
                    AddSegment(g, seg_name, origin, dest, cost)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return g

#Modification Version 2
# Add to Graph.py
from path import Path
import math

def EuclideanDistance(node1, node2):
    """Calculate Euclidean distance between two nodes"""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def FindReachableNodes(g, start_name):
    """Find all nodes reachable from start node"""
    start = g.find_node(start_name)
    if not start:
        return None
    
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        current = queue.pop(0)
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

"""def FindShortestPath(g, origin_name, destination_name):
    Path = _import_Path()
    origin = g.find_node(origin_name)
    destination = g.find_node(destination_name)
    if not origin or not destination:
        return None
    
    # Initialize open and closed lists
    open_paths = [Path(origin)]
    closed_nodes = set()
    
    while open_paths:
        # Find path with lowest estimated total cost
        min_cost = float('inf')
        best_path = None
        best_index = -1
        
        for i, path in enumerate(open_paths):
            last_node = path.GetLastNode()
            heuristic = EuclideanDistance(last_node, destination)
            total_estimate = path.cost + heuristic
            
            if total_estimate < min_cost:
                min_cost = total_estimate
                best_path = path
                best_index = i
        
        # Remove best path from open list
        current_path = open_paths.pop(best_index)
        last_node = current_path.GetLastNode()
        
        # Check if we've reached the destination
        if last_node == destination:
            return current_path
        
        # Add to closed set
        closed_nodes.add(last_node)
        
        # Explore neighbors
        for segment in g.segments.values():
            if segment.origin == last_node:
                neighbor = segment.destination
                
                # Skip if already in closed set
                if neighbor in closed_nodes:
                    continue
                
                # Check if neighbor is already in a path in open_paths
                found_better = False
                for existing_path in open_paths:
                    if existing_path.ContainsNode(neighbor):
                        # If existing path to neighbor has higher cost, remove it
                        if existing_path.CostToNode(neighbor) > current_path.cost + segment.cost:
                            open_paths.remove(existing_path)
                        else:
                            found_better = True
                        break
                
                # If no better path exists, add new path
                if not found_better:
                    new_path = current_path.Copy()
                    new_path.AddNodeToPath(neighbor, segment.cost)
                    open_paths.append(new_path)
    
    return None  # No path found"""
def FindShortestPath(g, origin_name, destination_name):
    Path = _import_Path()
    origin = g.find_node(origin_name)
    destination = g.find_node(destination_name)
    if not origin or not destination:
        return None
    
    # Initialize open and closed lists
    open_paths = [Path(origin)]
    closed_nodes = set()
    
    while open_paths:
        # Find path with lowest estimated total cost
        min_cost = float('inf')
        best_path = None
        best_index = -1
        
        for i, path in enumerate(open_paths):
            last_node = path.GetLastNode()
            heuristic = EuclideanDistance(last_node, destination)
            total_estimate = path.cost + heuristic
            
            if total_estimate < min_cost:
                min_cost = total_estimate
                best_path = path
                best_index = i
        
        # Remove best path from open list
        current_path = open_paths.pop(best_index)
        last_node = current_path.GetLastNode()
        
        # Check if we've reached the destination
        if last_node == destination:
            return current_path
        
        # Add to closed set
        closed_nodes.add(last_node)
        
        # Explore neighbors - THIS IS THE FIXED PART
        for segment in g.segments.values():
            if segment.origin == last_node:
                neighbor = segment.destination
                
                # Skip if already in closed set
                if neighbor in closed_nodes:
                    continue
                
                # Check if neighbor is already in a path in open_paths
                found_better = False
                for existing_path in open_paths:
                    if existing_path.GetLastNode() == neighbor:
                        # If existing path to neighbor has higher cost, remove it
                        if existing_path.cost > current_path.cost + segment.cost:
                            open_paths.remove(existing_path)
                        else:
                            found_better = True
                        break
                
                # If no better path exists, add new path
                if not found_better:
                    new_path = current_path.Copy()
                    new_path.AddNodeToPath(neighbor, segment.cost)
                    open_paths.append(new_path)
    
    return None  # No path found
    return current_path

def _import_Path():
    from path import Path
    return Path
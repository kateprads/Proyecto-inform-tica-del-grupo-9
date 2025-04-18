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

def AddSegment(g, name, nameOrigin, nameDestination, cost=1):
    origin = g.find_node(nameOrigin)
    destination = g.find_node(nameDestination)

    if origin is None or destination is None:
        return False

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
    
    # Dibujar segmentos
    for segment in g.segments.values():
        ax.plot([segment.origin.x, segment.destination.x],
                [segment.origin.y, segment.destination.y],
                'k-', alpha=0.5)
        # Etiqueta de costo
        mid_x = (segment.origin.x + segment.destination.x) / 2
        mid_y = (segment.origin.y + segment.destination.y) / 2
        ax.text(mid_x, mid_y, str(segment.cost), color='red')

    # Dibujar nodos
    for node in g.nodes.values():
        ax.plot(node.x, node.y, 'bo')
        ax.text(node.x, node.y, node.name, fontsize=12, ha='right')

    ax.grid(True)
    if ax is None:
        plt.show()

def PlotNode(g, node_name, ax=None):
    node = g.find_node(node_name)
    if node is None:
        return False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar todo el grafo primero
    Plot(g, ax)
    
    # Resaltar el nodo específico
    ax.plot(node.x, node.y, 'ro', markersize=10)
    ax.text(node.x, node.y, node.name, fontsize=12, ha='right', color='red')
    
    # Resaltar conexiones
    for segment in g.segments.values():
        if segment.origin == node or segment.destination == node:
            ax.plot([segment.origin.x, segment.destination.x],
                    [segment.origin.y, segment.destination.y],
                    'r-', linewidth=2)
    
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


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
        self.nodes = []
        self.segments = []

    def find_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None


def AddNode(g, n):
    if g.find_node(n.name) is not None:
        return False
    g.nodes.append(n)
    return True


def AddSegment(g, name, nameOrigin, nameDestination, cost=1):
    origin = g.find_node(nameOrigin)
    destination = g.find_node(nameDestination)

    if origin is None or destination is None:
        return False

    segment = Segment(name, origin, destination, cost)
    g.segments.append(segment)
    origin.neighbors.append(destination)
    return True


def GetClosest(g, x, y):
    closest_node = None
    min_distance = float('inf')

    for node in g.nodes:
        distance = math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node


def Plot(g,ax=None):
    plt.figure(figsize=(10, 8))

    # Draw segments
    for segment in g.segments:
        plt.plot([segment.origin.x, segment.destination.x],
                 [segment.origin.y, segment.destination.y],
                 'k-', alpha=0.5)
        # Add cost label at midpoint
        mid_x = (segment.origin.x + segment.destination.x) / 2
        mid_y = (segment.origin.y + segment.destination.y) / 2
        plt.text(mid_x, mid_y, str(segment.cost), color='red')

    # Draw nodes
    for node in g.nodes:
        plt.plot(node.x, node.y, 'bo')
        plt.text(node.x, node.y, node.name, fontsize=12, ha='right')

    plt.grid(True)
    plt.show(block=False)
    plt.show()


def PlotNode(g, nameOrigin):
    origin = g.find_node(nameOrigin)
    if origin is None:
        return False

    plt.figure(figsize=(10, 8))

    # Draw all segments first (gray)
    for segment in g.segments:
        plt.plot([segment.origin.x, segment.destination.x],
                 [segment.origin.y, segment.destination.y],
                 'k-', color='gray', alpha=0.3)

    # Highlight origin node and its connections
    for segment in g.segments:
        if segment.origin == origin or segment.destination == origin:
            # Red for segments connected to origin
            plt.plot([segment.origin.x, segment.destination.x],
                     [segment.origin.y, segment.destination.y],
                     'r-', linewidth=2)
            # Add cost label at midpoint
            mid_x = (segment.origin.x + segment.destination.x) / 2
            mid_y = (segment.origin.y + segment.destination.y) / 2
            plt.text(mid_x, mid_y, str(segment.cost), color='red')

    # Draw all nodes
    for node in g.nodes:
        if node == origin:
            # Origin node in blue
            plt.plot(node.x, node.y, 'bo', markersize=10)
        elif node in origin.neighbors:
            # Neighbors in green
            plt.plot(node.x, node.y, 'go', markersize=8)
        else:
            # Others in gray
            plt.plot(node.x, node.y, 'o', color='gray', markersize=6)

        plt.text(node.x, node.y, node.name, fontsize=12, ha='right')

    plt.grid(True)
    plt.show(block=False)
    plt.show()
    return True

def ReadGraphFromFile(filename):
    """
    Llegeix un graf des d'un fitxer de text
    Format del fitxer:
    NODE nom_node x y
    SEGMENT nom_segment node_origen node_destí [cost]
    """
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

    except FileNotFoundError:
        print(f"Error: No s'ha trobat el fitxer {filename}")
        return None
    except Exception as e:
        print(f"Error en llegir el fitxer: {e}")
        return None

    return g

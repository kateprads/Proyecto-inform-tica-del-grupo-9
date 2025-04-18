import unittest
from Graph import Graph, Node, AddNode, AddSegment, Plot, PlotNode, GetClosest
import matplotlib.pyplot as plt

def CreateGraph_1():
    """Creates the exact graph shown in your reference image"""
    G = Graph()
    
    # Add nodes with coordinates matching your image
    nodes = [
        ("A", 1, 20), ("B", 8, 17), ("C", 15, 20),
        ("D", 18, 15), ("E", 2, 4), ("F", 6, 5),
        ("G", 12, 12), ("H", 10, 3), ("I", 19, 1),
        ("J", 13, 5), ("K", 3, 15), ("L", 4, 10)
    ]
    
    for name, x, y in nodes:
        AddNode(G, Node(name, x, y))
    
    # Add edges (segments) matching your image
    edges = [
        ("AB", "A", "B"), ("AE", "A", "E"), ("AK", "A", "K"),
        ("BA", "B", "A"), ("BC", "B", "C"), ("BF", "B", "F"),
        ("BL", "B", "L"), ("CB", "C", "B"), ("CD", "C", "D"),
        ("CG", "C", "G"), ("DC", "D", "C"), ("EA", "E", "A"),
        ("EL", "E", "L"), ("FB", "F", "B"), ("FJ", "F", "J"),
        ("FL", "F", "L"), ("GC", "G", "C"), ("GJ", "G", "J"),
        ("GK", "G", "K"), ("GL", "G", "L"), ("HG", "H", "G"),
        ("HJ", "H", "J"), ("HL", "H", "L"), ("IH", "I", "H"),
        ("JI", "J", "I"), ("JF", "J", "F"), ("JG", "J", "G"),
        ("JH", "J", "H"), ("KA", "K", "A"), ("KG", "K", "G"),
        ("KL", "K", "L"), ("LE", "L", "E"), ("LF", "L", "F"),
        ("LG", "L", "G"), ("LH", "L", "H"), ("LK", "L", "K")
    ]
    
    for name, origin, dest in edges:
        AddSegment(G, name, origin, dest)
    
    return G

def CreateGraph_2():
    """Creates a custom star-shaped graph with a central hub"""
    G = Graph()
    
    # Central hub
    AddNode(G, Node("HUB", 10, 10))
    
    # Outer nodes
    outer_nodes = [
        ("N", 10, 20), ("NE", 15, 17), ("E", 20, 10),
        ("SE", 15, 3), ("S", 10, 0), ("SW", 5, 3),
        ("W", 0, 10), ("NW", 5, 17)
    ]
    
    for name, x, y in outer_nodes:
        AddNode(G, Node(name, x, y))
        # Connect each to hub
        AddSegment(G, f"HUB-{name}", "HUB", name)
        AddSegment(G, f"{name}-HUB", name, "HUB")
    
    # Add some cross connections
    AddSegment(G, "N-NE", "N", "NE")
    AddSegment(G, "NE-E", "NE", "E")
    AddSegment(G, "E-SE", "E", "SE")
    AddSegment(G, "SE-S", "SE", "S")
    AddSegment(G, "S-SW", "S", "SW")
    AddSegment(G, "SW-W", "SW", "W")
    AddSegment(G, "W-NW", "W", "NW")
    AddSegment(G, "NW-N", "NW", "N")
    
    return G

def test_graph():
    """Tests and displays both graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Test CreateGraph_1
    G1 = CreateGraph_1()
    Plot(G1, ax1)
    ax1.set_title("Original Graph (CreateGraph_1)")
    
    # Highlight node C and neighbors
    neighbors_of_C = set()
    for seg in G1.segments.values():
        if seg.origin.name == "C":
            neighbors_of_C.add(seg.destination.name)
        elif seg.destination.name == "C":
            neighbors_of_C.add(seg.origin.name)
    
    PlotNode(G1, "C", ax1, color="red")
    for node in neighbors_of_C:
        PlotNode(G1, node, ax1, color="orange")
    
    # Test GetClosest
    closest = GetClosest(G1, 5, 18)
    ax1.plot(5, 18, 'x', color="purple", markersize=12)
    ax1.text(5, 18.5, f"Closest to (5,18)\nis {closest.name}", ha='center')
    
    # Test CreateGraph_2
    G2 = CreateGraph_2()
    Plot(G2, ax2)
    ax2.set_title("Custom Star Graph (CreateGraph_2)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_graph()

# Bloque de pruebas (comentado para evitar ejecución al importar)
if __name__ == "__main__":
    print("=== Prueba de CreateGraph_1 ===")
    G1 = CreateGraph_1()
    Plot(G1)
    
    print("\n=== Prueba de CreateGraph_2 ===")
    G2 = CreateGraph_2()
    Plot(G2)
    
    print("\n=== Prueba de GetClosest ===")
    closest = GetClosest(G1, 15, 5)
    print(f"Nodo más cercano a (15,5): {closest.name} (debería ser J)")
    
    closest = GetClosest(G1, 8, 19)
    print(f"Nodo más cercano a (8,19): {closest.name} (debería ser B)")

class TestGraphAlgorithms(unittest.TestCase):
    def setUp(self):
        """Create a simple graph for testing"""
        self.graph = Graph()
        
        # Create test nodes
        self.nodeA = Node("A", 0, 0)
        self.nodeB = Node("B", 1, 1)
        self.nodeC = Node("C", 1, 0)
        
        # Add nodes to graph
        self.graph.AddNode(self.nodeA)
        self.graph.AddNode(self.nodeB)
        self.graph.AddNode(self.nodeC)
        
        # Create connections with costs
        self.graph.add_edge(self.nodeA, self.nodeB, 2.0)  # A-B costs 2.0
        self.graph.add_edge(self.nodeB, self.nodeC, 3.0)  # B-C costs 3.0

    def test_reachable_nodes(self):
        """Test finding all reachable nodes from a starting node"""
        # From A, should reach A, B, C
        reachable = self.graph.find_reachable_nodes(self.nodeA)
        self.assertEqual(len(reachable), 3)
        
        # From B, should reach B, A, C
        reachable = self.graph.find_reachable_nodes(self.nodeB)
        self.assertEqual(len(reachable), 3)
        
        # If we add a disconnected node D
        nodeD = Node("D", 2, 2)
        self.graph.add_node(nodeD)
        # From D, should only reach D itself
        reachable = self.graph.find_reachable_nodes(nodeD)
        self.assertEqual(len(reachable), 1)

    def test_shortest_path(self):
        """Test finding the shortest path between two nodes"""
        # A to C should be A-B-C with cost 5.0
        path = self.graph.find_shortest_path(self.nodeA, self.nodeC)
        self.assertIsNotNone(path)
        self.assertEqual(path.cost, 5.0)
        self.assertEqual(len(path.nodes), 3)
        
        # A to A should be just A with cost 0
        path = self.graph.find_shortest_path(self.nodeA, self.nodeA)
        self.assertIsNotNone(path)
        self.assertEqual(path.cost, 0)
        self.assertEqual(len(path.nodes), 1)
        
        # Test unreachable path
        nodeD = Node("D", 2, 2)
        self.graph.add_node(nodeD)
        path = self.graph.find_shortest_path(self.nodeA, nodeD)
        self.assertIsNone(path)

if __name__ == '__main__':
    unittest.main()


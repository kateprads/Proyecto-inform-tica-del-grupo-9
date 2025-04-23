import unittest
from Graph import Graph
from path import Path

class TestGraphFunctions(unittest.TestCase):
    """Test cases for the new Graph class functionality."""
    
    def setUp(self):
        """Set up a graph for testing as shown in the example."""
        self.graph = Graph()
        
        # Add nodes as shown in the figure from the instructions
        nodes = {
            "A": (0, 0),
            "B": (1, 1),
            "C": (2, 0),
            "D": (3, 1),
            "E": (4, 0),
            "F": (5, 1),
            "K": (2, 2)}
        
        for node, (x, y) in nodes.items():
            self.graph.add_node(node, x, y)
        
        # Add edges as shown in the figure
        edges = [
            ("A", "B", 1.5),  # Specified weight
            ("B", "C", 1.5),
            ("C", "D", 1.5),
            ("D", "E", 1.5),
            ("E", "F", 1.5),
            ("B", "K", 1.5),
            ("K", "D", 1.5)
        ]
        
        for node1, node2, weight in edges:
            self.graph.add_edge(node1, node2, weight)
    
    def test_find_shortest_path(self):
        """Test finding the shortest path using the A* algorithm."""
        # Test path from A to F (should go through B, C, D, E)
        path = self.graph.find_shortest_path("A", "F")
        self.assertIsNotNone(path)
        self.assertEqual(path.nodes, ["A", "B", "C", "D", "E", "F"])
        
        # Check the total cost (should be 7.5 = 5 * 1.5)
        self.assertEqual(path.costs[-1], 7.5)
        
        # Test with a path that should go through K
        # Make the C-D edge expensive to force path through K
        self.graph.edges["C"]["D"] = 10.0
        self.graph.edges["D"]["C"] = 10.0
        
        path = self.graph.find_shortest_path("A", "F")
        self.assertIsNotNone(path)
        # Now the path should go through K
        self.assertEqual(path.nodes, ["A", "B", "K", "D", "E", "F"])
        
        # Check the total cost (A-B=1.5, B-K=1.5, K-D=1.5, D-E=1.5, E-F=1.5)
        self.assertEqual(path.costs[-1], 7.5)
    
    def test_no_path(self):
        """Test when there is no path between nodes."""
        # Add a disconnected node
        self.graph.add_node("Z", 10, 10)
        
        path = self.graph.find_shortest_path("A", "Z")
        self.assertIsNone(path)
    
    def test_find_reachable_nodes(self):
        """Test finding all reachable nodes from a starting node."""
        # All nodes should be reachable from A
        reachable = self.graph.find_reachable_nodes("A")
        self.assertEqual(set(reachable), {"A", "B", "C", "D", "E", "F", "K"})
        
        # Test with a disconnected node
        self.graph.add_node("Z", 10, 10)
        reachable = self.graph.find_reachable_nodes("A")
        self.assertNotIn("Z", reachable)
        
        reachable = self.graph.find_reachable_nodes("Z")
        self.assertEqual(reachable, ["Z"])

if __name__ == "__main__":
    unittest.main()

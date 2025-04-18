import unittest
from path import Path
from .test_node import Node  # Assuming you have a Node class

class TestPath(unittest.TestCase):
    def setUp(self):
        self.nodeA = Node("A", 0, 0)
        self.nodeB = Node("B", 1, 1)
        self.nodeC = Node("C", 2, 2)
    
    def test_path_creation(self):
        path = Path(self.nodeA)
        self.assertEqual(len(path.nodes), 1)
        self.assertEqual(path.cost, 0)
    
    def test_add_node(self):
        path = Path(self.nodeA)
        path.add_node(self.nodeB, 5)
        self.assertEqual(len(path.nodes), 2)
        self.assertEqual(path.cost, 5)
    
    # More test cases...

if __name__ == '__main__':
    unittest.main()
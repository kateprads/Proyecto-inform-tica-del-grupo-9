#version 2 new file
# test_path.py
from Graph import Graph, Node, AddNode, AddSegment
from path import Path, PlotPath

def test_path_functions():
    print("=== Testing Path Functions ===")
    
    # Create a simple graph
    g = Graph()
    n1 = Node("A", 0, 0)
    n2 = Node("B", 1, 1)
    n3 = Node("C", 2, 0)
    AddNode(g, n1)
    AddNode(g, n2)
    AddNode(g, n3)
    AddSegment(g, "AB", "A", "B", 1.5)
    AddSegment(g, "BC", "B", "C", 2.0)
    AddSegment(g, "AC", "A", "C", 4.0)
    
    # Test Path class
    p = Path(n1)
    print("Initial path:", p)
    
    # Test AddNodeToPath
    p.AddNodeToPath(n2, 1.5)
    print("After adding B:", p)
    
    p.AddNodeToPath(n3, 2.0)
    print("After adding C:", p)
    
    # Test ContainsNode
    print("Contains A:", p.ContainsNode(n1))  # True
    print("Contains D:", p.ContainsNode(Node("D", 3, 3)))  # False
    
    # Test CostToNode
    print("Cost to B:", p.CostToNode(n2))  # 1.5
    print("Cost to C:", p.CostToNode(n3))  # 3.5
    print("Cost to D:", p.CostToNode(Node("D", 3, 3)))  # -1
    
    # Test PlotPath
    print("Plotting path...")
    PlotPath(g, p)
    print("Path tests passed!\n")

if __name__ == "__main__":
    test_path_functions()
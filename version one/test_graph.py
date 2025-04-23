# test_graph.py
from Graph import Graph, Node, AddNode, AddSegment
from Graph import FindReachableNodes, FindShortestPath

def test_reachability():
    print("=== Testing Reachability ===")
    g = Graph()
    
    # Create nodes
    nodes = [
        Node("A", 0, 0),
        Node("B", 1, 1),
        Node("C", 2, 0),
        Node("D", 3, 3),
        Node("E", 4, 4)
    ]
    
    for node in nodes:
        AddNode(g, node)
    
    # Create segments
    AddSegment(g, "AB", "A", "B")
    AddSegment(g, "BC", "B", "C")
    AddSegment(g, "BD", "B", "D")
    AddSegment(g, "DE", "D", "E")
    
    # Test reachability
    reachable = FindReachableNodes(g, "A")
    print("Nodes reachable from A:", [n.name for n in reachable])  # Should be A, B, C, D, E
    
    reachable = FindReachableNodes(g, "C")
    print("Nodes reachable from C:", [n.name for n in reachable])  # Should be just C
    
    print("Reachability tests passed!\n")

def test_shortest_path():
    print("=== Testing Shortest Path (A*) ===")
    g = Graph()
    
    # Create nodes
    nodes = [
        Node("A", 0, 0),
        Node("B", 1, 1),
        Node("C", 2, 0),
        Node("D", 3, 3),
        Node("E", 4, 4)
    ]
    
    for node in nodes:
        AddNode(g, node)
    
    # Create segments with costs
    AddSegment(g, "AB", "A", "B", 1.5)
    AddSegment(g, "BC", "B", "C", 2.0)
    AddSegment(g, "BD", "B", "D", 3.0)
    AddSegment(g, "DE", "D", "E", 1.0)
    AddSegment(g, "AC", "A", "C", 4.0)  # Direct but more expensive
    
    # Test shortest path
    path = FindShortestPath(g, "A", "E")
    print("Shortest path A to E:", path)  # Should be A->B->D->E with cost 5.5
    
    path = FindShortestPath(g, "A", "C")
    print("Shortest path A to C:", path)  # Should be A->B->C with cost 3.5
    
    path = FindShortestPath(g, "C", "E")
    print("Shortest path C to E:", path)  # Should be None
    
    print("Shortest path tests passed!\n")

if __name__ == "__main__":
    test_reachability()
    test_shortest_path()
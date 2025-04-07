#test_segment.py that uses the Segment class and creates nodes to test it:

class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Segment:
    def __init__(self, name, origin_node, destination_node):
        self.name = name
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.cost = ((destination_node.x - origin_node.x) ** 2 + (destination_node.y - origin_node.y) ** 2) ** 0.5

# Create 3 nodes with x, y coordinates

def main():
    node1 = Node(1, 1)
    node2 = Node(3, 4)
    node3 = Node(6, 8)

    segment1 = Segment("S1", node1, node2)
    segment2 = Segment("S2", node2, node3)

    print("Segment 1")
    print(f"Name: {segment1.name}")
    print(f"Origin: ({segment1.origin_node.x}, {segment1.origin_node.y})")
    print(f"Destination: ({segment1.destination_node.x}, {segment1.destination_node.y})")
    print(f"Cost: {segment1.cost:.2f}")
    
    print("\nSegment 2")
    print(f"Name: {segment2.name}")
    print(f"Origin: ({segment2.origin_node.x}, {segment2.origin_node.y})")
    print(f"Destination: ({segment2.destination_node.x}, {segment2.destination_node.y})")
    print(f"Cost: {segment2.cost:.2f}")

if __name__ == "__main__":
    main()
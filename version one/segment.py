class segemnt:
    def __init__(self,name,origin_node,destination_node):
        self.name= name
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.cost = ((destination_node.x - origin_node.x) ** 2 + (destination_node.y - origin_node.y) ** 2) ** 0.5
#Calculates the cost as the Euclidean distance between the two nodes



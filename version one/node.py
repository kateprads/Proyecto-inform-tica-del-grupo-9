import math

class Node:
    def __init__(self, name: str, coordinate_x: float, coordinate_y: float):
        self.name = name
        self.coordinate_x = coordinate_x
        self.coordinate_y = coordinate_y
        self.neighbors = [] #List to store neighboring nodes
        
    def Addneighbor(n1,n2): # # Adds node n2 as a neighbor of node n1, if not already present
        if n2 not in n1.neighbors:
            n1.neighbors.append(n2)
            return True
        return False
    
    
    def euclidean_distance(n1,n2): #distance= square root (x2-x1)^2 + (y2-y1)^2, between n1 and n2
        dx = n2.coordinate_x - n1.coordinate_x
        dy = n2.coordinate_y - n1.coordinate_y
        return math.sqrt(dx ** 2 + dy ** 2)

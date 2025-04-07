from xml.dom import Node
from node import * # # Import everything from the node module

## Create nodes n1 and n2
n1 = Node ('aaa', 0, 0)
n2 = Node ('bbb', 3, 4)

## Test the distance between n1 and n2
print (n1.euclidean_distance(n2))

## Add n2 as a neighbor of n1
print (n1.add_neighbor(n2))

# Try adding n2 again as a neighbor of n1
print (n1.add_neighbor(n2))

## Print the internal dictionary of n1 (this shows all its attributes)
print (n1.__dict__)

## Print the internal dictionaries of n1's neighbors
for n in n1.neighbors:
 print ( n.__dict__)

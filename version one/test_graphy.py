# test_graphy.py
from platform import node
import unittest
from Graph import Graph, Node, AddNode, AddSegment, Plot, PlotNode, GetClosest
from Graph import Graph
from test_node import Node

def CreateGraph_1():
    G = Graph()
    # Añadir todos los nodos
    AddNode(G, Node("A", 1, 20))
    AddNode(G, Node("B", 8, 17))
    AddNode(G, Node("C", 15, 20))
    AddNode(G, Node("D", 18, 15))
    AddNode(G, Node("E", 2, 4))
    AddNode(G, Node("F", 6, 5))
    AddNode(G, Node("G", 12, 12))
    AddNode(G, Node("H", 10, 3))
    AddNode(G, Node("I", 19, 1))
    AddNode(G, Node("J", 13, 5))
    AddNode(G, Node("K", 3, 15))
    AddNode(G, Node("L", 4, 10))
    
    # Añadir todos los segmentos
    AddSegment(G, "AB", "A", "B")
    AddSegment(G, "AE", "A", "E")
    AddSegment(G, "AK", "A", "K")
    AddSegment(G, "BA", "B", "A")
    AddSegment(G, "BC", "B", "C")
    AddSegment(G, "BF", "B", "F")
    AddSegment(G, "BK", "B", "K")
    AddSegment(G, "BG", "B", "G")
    AddSegment(G, "CD", "C", "D")
    AddSegment(G, "CG", "C", "G")
    AddSegment(G, "DG", "D", "G")
    AddSegment(G, "DH", "D", "H")
    AddSegment(G, "DI", "D", "I")
    AddSegment(G, "EF", "E", "F")
    AddSegment(G, "FL", "F", "L")
    AddSegment(G, "GB", "G", "B")
    AddSegment(G, "GF", "G", "F")
    AddSegment(G, "GH", "G", "H")
    AddSegment(G, "ID", "I", "D")
    AddSegment(G, "IJ", "I", "J")
    AddSegment(G, "JI", "J", "I")
    AddSegment(G, "KA", "K", "A")
    AddSegment(G, "KL", "K", "L")
    AddSegment(G, "LK", "L", "K")
    AddSegment(G, "LF", "L", "F")
    
    return G

def CreateGraph_2():
    """Grafo personalizado con 4 nodos"""
    G = Graph()
    # Añadir nodos
    AddNode(G, Node("M", 5, 5))
    AddNode(G, Node("N", 15, 5))
    AddNode(G, Node("O", 10, 15))
    AddNode(G, Node("P", 20, 20))
    
    # Añadir segmentos con costos
    AddSegment(G, "MN", "M", "N", 2)
    AddSegment(G, "MO", "M", "O", 3)
    AddSegment(G, "NO", "N", "O", 4)
    AddSegment(G, "OP", "O", "P", 1)
    AddSegment(G, "NP", "N", "P", 5)
    
    return G

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


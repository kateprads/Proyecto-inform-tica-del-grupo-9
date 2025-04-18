from Graph import Node, Segment  # Importar desde Graph.py

def main():
    # Crear nodos
    node1 = Node("N1", 1, 1)
    node2 = Node("N2", 3, 4)
    node3 = Node("N3", 6, 8)

    # Crear segmentos
    segment1 = Segment("S1", node1, node2)
    segment2 = Segment("S2", node2, node3)

    # Mostrar información
    print("=== Segmento 1 ===")
    print(f"Nombre: {segment1.name}")
    print(f"Origen: {segment1.origin.name} ({segment1.origin.x}, {segment1.origin.y})")
    print(f"Destino: {segment1.destination.name} ({segment1.destination.x}, {segment1.destination.y})")
    print(f"Costo: {segment1.cost:.2f}")
    
    print("\n=== Segmento 2 ===")
    print(f"Nombre: {segment2.name}")
    print(f"Origen: {segment2.origin.name} ({segment2.origin.x}, {segment2.origin.y})")
    print(f"Destino: {segment2.destination.name} ({segment2.destination.x}, {segment2.destination.y})")
    print(f"Costo: {segment2.cost:.2f}")

if __name__ == "__main__":
    main()
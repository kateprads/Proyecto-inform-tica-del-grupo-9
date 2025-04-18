from Graph import Node  # Importar desde Graph.py

def main():
    # Crear nodos de prueba
    n1 = Node("A", 0, 0)
    n2 = Node("B", 3, 4)
    
    # Probar distancia euclidiana
    distance = ((n2.x - n1.x)**2 + (n2.y - n1.y)**2)**0.5
    print(f"Distancia entre {n1.name} y {n2.name}: {distance:.2f}")
    
    # Probar añadir vecinos
    n1.neighbors.append(n2)
    print(f"Vecinos de {n1.name}: {[n.name for n in n1.neighbors]}")
    
    # Mostrar atributos
    print(f"\nDetalles de {n1.name}:")
    print(f"Nombre: {n1.name}")
    print(f"Coordenadas: ({n1.x}, {n1.y})")
    print(f"Vecinos: {[n.name for n in n1.neighbors]}")

if __name__ == "__main__":
    main()
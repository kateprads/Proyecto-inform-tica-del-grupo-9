# visualize_graph.py
from Graph import ReadGraphFromFile, Plot


def main():
    # Demanar a l'usuari el nom del fitxer o posar-lo directament
    filename = input("Introdueix el nom del fitxer del graf (per defecte: my_graph.txt): ") or "my_graph.txt"

    # Llegir el graf des del fitxer
    print(f"Llegint el graf des de {filename}...")
    my_graph = ReadGraphFromFile(filename)

    if my_graph:
        print(f"Gràfic llegit correctament amb {len(my_graph.nodes)} nodes i {len(my_graph.segments)} segments.")
        Plot(my_graph)  # Mostrar el gràfic
    else:
        print("No s'ha pogut llegir el gràfic.")


if __name__ == "__main__":
    main()
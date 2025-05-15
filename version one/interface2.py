import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from Graph import Graph, Node, AddNode, AddSegment, ReadGraphFromFile, Plot, PlotNode, GetClosest, FindReachableNodes, FindShortestPath
from airspaceLoader import AirspaceLoader
import graphConverter
from path import PlotPath
import heapq


class GraphInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizador de Grafos Avanzado")
        self.geometry("1100x850")
        self.graph = None
        
        # Configurar tema ttk
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Paleta de colores
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#4a7a8c"
        
        # Configurar estilos personalizados
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TButton", 
                            background=self.accent_color, 
                            foreground="white",
                            font=("Helvetica", 10, "bold"),
                            padding=5)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Frame de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Título
        title_label = ttk.Label(
            control_frame, 
            text="VISUALIZADOR DE GRAFOS", 
            font=("Helvetica", 20, "bold"))
        title_label.pack(pady=10)
        
        # Botones principales
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        button_data = [
            ("Grafo de Ejemplo", self.load_example_graph),
            ("Grafo Personalizado", self.load_custom_graph),
            ("Cargar desde Archivo", self.load_from_file),
            ("Guardar Grafo", self.save_graph)]
        
        for text, command in button_data:
            btn = ttk.Button(
                button_frame, 
                text=text, 
                command=command,
                style="TButton"
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Frame de búsqueda y edición
        edit_frame = ttk.Frame(control_frame)
        edit_frame.pack(pady=10)
        
        # Búsqueda de nodos
        search_label = ttk.Label(
            edit_frame, 
            text="Buscar Nodo:",
            font=("Helvetica", 11))
        search_label.pack(side=tk.LEFT, padx=5)
        
        self.node_entry = ttk.Entry(
            edit_frame, 
            font=("Helvetica", 11), 
            width=20)
        self.node_entry.pack(side=tk.LEFT, padx=5)
        
        search_btn = ttk.Button(
            edit_frame, 
            text="Mostrar Vecinos", 
            command=self.show_node_neighbors,
            style="TButton")
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Botones para edición
        edit_buttons = [
            ("Añadir Nodo", self.add_node),
            ("Añadir Segmento", self.add_segment),
            ("Eliminar Nodo", self.delete_node),
            ("Nodo más Cercano", self.find_closest_node),
            ("Nodos Alcanzables", self.show_reachable_nodes),
            ("Camino más Corto", self.show_shortest_path)]
            
        
        for text, command in edit_buttons:
            btn = ttk.Button(
                edit_frame, 
                text=text, 
                command=command,
                style="TButton")
            btn.pack(side=tk.LEFT, padx=5)
        
        # Área del gráfico
        graph_frame = tk.Frame(main_frame, bg="white")
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.text(
            0.5, 0.5, 
            "Seleccione una opción para visualizar un grafo", 
            ha="center", va="center", 
            fontsize=12, color="gray"
        )
        self.ax.axis("off")
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar de Matplotlib
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def refresh_graph(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        if self.graph and hasattr(self.graph, 'nodes'):
            Plot(self.graph, self.ax)
        else:
            self.ax.text(0.5, 0.5, "Seleccione una opción para visualizar un grafo", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.axis("off")
        self.canvas.draw()
    
    def load_example_graph(self):
        try:
            from test_graphy import CreateGraph_1
            self.graph = CreateGraph_1()
            self.refresh_graph()
            messagebox.showinfo("Éxito", "Grafo de ejemplo cargado correctamente")
        except ImportError as e:
            messagebox.showerror("Error", 
                            f"No se pudo importar CreateGraph_1:\n"
                            f"1. Verifica que test_graphy.py esté en la misma carpeta\n"
                            f"2. Revisa que tenga la función CreateGraph_1()\n"
                            f"Error detallado: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado:\n{str(e)}")

    def load_custom_graph(self):
        try:
            from test_graphy import CreateGraph_2
            self.graph = CreateGraph_2()
            self.refresh_graph()
            messagebox.showinfo("Éxito", "Grafo personalizado cargado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el grafo personalizado:\n{str(e)}")
        
    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de grafo",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if file_path:
            try:
                self.graph = ReadGraphFromFile(file_path)
                if self.graph:
                    self.refresh_graph()
                    messagebox.showinfo("Éxito", f"Grafo cargado correctamente\nNodos: {len(self.graph.nodes)}\nSegmentos: {len(self.graph.segments)}")
                else:
                    messagebox.showerror("Error", "El archivo no contiene un grafo válido")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def save_graph(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "No hay ningún grafo cargado")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar grafo",
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("NODES\n")
                    for node in self.graph.nodes.values():
                        f.write(f"{node.name} {node.x} {node.y}\n")
                    
                    f.write("\nSEGMENTS\n")
                    for segment in self.graph.segments.values():
                        f.write(f"{segment.name} {segment.origin.name} {segment.destination.name} {segment.cost}\n")
                
                messagebox.showinfo("Éxito", f"Grafo guardado correctamente en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el grafo:\n{str(e)}")

    def show_node_neighbors(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar un grafo")
            return
            
        node_name = self.node_entry.get().strip()
        if not node_name:
            messagebox.showwarning("Advertencia", "Ingrese el nombre de un nodo")
            return
            
        try:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            Plot(self.graph, self.ax)
            
            if not PlotNode(self.graph, node_name, self.ax):
                messagebox.showerror("Error", f"Nodo '{node_name}' no encontrado")
                self.refresh_graph()
                return
                
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar nodo:\n{str(e)}")
            self.refresh_graph()
    
    def add_node(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return
            
        node_name = simpledialog.askstring("Añadir Nodo", "Nombre del nuevo nodo:")
        if not node_name:
            return
            
        if node_name in self.graph.nodes:
            messagebox.showerror("Error", f"El nodo '{node_name}' ya existe")
            return
            
        try:
            x = float(simpledialog.askstring("Coordenadas", "Coordenada X:"))
            y = float(simpledialog.askstring("Coordenadas", "Coordenada Y:"))
            
            AddNode(self.graph, Node(node_name, x, y))
            self.refresh_graph()
            messagebox.showinfo("Éxito", f"Nodo '{node_name}' añadido correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo añadir el nodo:\n{str(e)}")
    
    
    def add_segment(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return
            
        origin = simpledialog.askstring("Añadir Segmento", "Nodo origen:")
        if not origin or origin not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo origen '{origin}' no válido")
            return
            
        destination = simpledialog.askstring("Añadir Segmento", "Nodo destino:")
        if not destination or destination not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo destino '{destination}' no válido")
            return
            
        try:
            segment_name = f"{origin}{destination}"
            AddSegment(self.graph, segment_name, origin, destination)
            self.refresh_graph()
            messagebox.showinfo("Éxito", f"Segmento '{segment_name}' añadido correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo añadir el segmento:\n{str(e)}")

    
    def delete_node(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return

        node_name = simpledialog.askstring("Eliminar Nodo", "Nombre del nodo a eliminar:")
        if not node_name:
            return

        if node_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo '{node_name}' no encontrado")
            return

        try:
            segments_to_remove = [
                seg_name for seg_name, seg in self.graph.segments.items()
                if seg.origin.name == node_name or seg.destination.name == node_name]
            
            for seg_name in segments_to_remove:
                del self.graph.segments[seg_name]
            
            del self.graph.nodes[node_name]
            
            self.refresh_graph()
            messagebox.showinfo("Éxito", f"Nodo '{node_name}' y {len(segments_to_remove)} segmentos eliminados")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el nodo:\n{str(e)}")
    
    def find_closest_node(self):
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return
            
        try:
            x = float(simpledialog.askstring("Coordenadas", "Coordenada X:"))
            y = float(simpledialog.askstring("Coordenadas", "Coordenada Y:"))
            
            closest_node = GetClosest(self.graph, x, y)
            if closest_node:
                messagebox.showinfo("Nodo más cercano", 
                                  f"El nodo más cercano a ({x}, {y}) es:\n"
                                  f"Nombre: {closest_node.name}\n"
                                  f"Coordenadas: ({closest_node.x}, {closest_node.y})")
                
                self.figure.clear()
                self.ax = self.figure.add_subplot(111)
                Plot(self.graph, self.ax)
                PlotNode(self.graph, closest_node.name, self.ax)
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo encontrar el nodo más cercano:\n{str(e)}")
#HERE ADDITIONS V2
    def show_reachable_nodes(self):
        """V2: Show all reachable nodes from a starting node"""
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return
        
        start = simpledialog.askstring("Nodos Alcanzables", "Nodo de inicio:")
        if not start or start not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo de inicio '{start}' no válido")
            return
        
        try:
            reachable = FindReachableNodes(self.graph, start)
            if not reachable:
                messagebox.showinfo("Resultado", f"No hay nodos alcanzables desde {start}")
                return
            
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            Plot(self.graph, self.ax)
            
            for node in reachable:
                self.ax.plot(node.x, node.y, 'go', markersize=10)
                self.ax.text(node.x, node.y, node.name, fontsize=12, ha='right')
            
            start_node = self.graph.find_node(start)
            self.ax.plot(start_node.x, start_node.y, 'ro', markersize=12)
            
            messagebox.showinfo("Resultado", 
                              f"Nodos alcanzables desde {start}: {len(reachable)}\n"
                              f"Incluyendo: {', '.join([n.name for n in reachable])}")
            
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron encontrar nodos alcanzables:\n{str(e)}")
            self.refresh_graph()

    def show_shortest_path(self):
        """V2: Show shortest path between two nodes using A*"""
        if not self.graph:
            messagebox.showwarning("Advertencia", "Primero debe cargar o crear un grafo")
            return
        
        start = simpledialog.askstring("Camino más corto", "Nodo de inicio:")
        if not start or start not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo de inicio '{start}' no válido")
            return
        
        end = simpledialog.askstring("Camino más corto", "Nodo de destino:")
        if not end or end not in self.graph.nodes:
            messagebox.showerror("Error", f"Nodo de destino '{end}' no válido")
            return
        
        try:
            path = FindShortestPath(self.graph, start, end)
            if not path:
                messagebox.showinfo("Resultado", f"No hay camino entre {start} y {end}")
                return
            
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            Plot(self.graph, self.ax)
            
            PlotPath(self.graph, path, self.ax)
            
            path_names = " → ".join([node.name for node in path.nodes])
            messagebox.showinfo("Camino más corto", 
                              f"Camino: {path_names}\n"
                              f"Costo total: {path.cost:.2f}")
            
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo calcular el camino:\n{str(e)}")
            self.refresh_graph()
#versión 3            

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from airSpace import AirSpace
from navPoint import NavPoint
from navSegment import NavSegment
from navAirport import NavAirport
import os

class AirspaceVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Airspace Visualization Tool - Version 3")
        self.geometry("1200x900")
        self.current_airspace = None
        self.graph = None
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.configure(background='#f0f0f0')
        
        self.create_widgets()
        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Catalunya", command=lambda: self.load_airspace("Catalunya"))
        file_menu.add_command(label="Load España", command=lambda: self.load_airspace("España"))
        file_menu.add_command(label="Load Europe", command=lambda: self.load_airspace("Europe"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Show Neighbors", command=self.show_neighbors)
        analysis_menu.add_command(label="Show Reachable Nodes", command=self.show_reachable_nodes)
        analysis_menu.add_command(label="Find Shortest Path", command=self.find_shortest_path)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        self.config(menu=menubar)

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Node selection
        node_frame = ttk.Frame(control_frame)
        node_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(node_frame, text="Node:").pack(side=tk.LEFT)
        self.node_entry = ttk.Entry(node_frame, width=20)
        self.node_entry.pack(side=tk.LEFT, padx=5)

        # Path selection
        path_frame = ttk.Frame(control_frame)
        path_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(path_frame, text="From:").pack(side=tk.LEFT)
        self.origin_entry = ttk.Entry(path_frame, width=15)
        self.origin_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(path_frame, text="To:").pack(side=tk.LEFT)
        self.dest_entry = ttk.Entry(path_frame, width=15)
        self.dest_entry.pack(side=tk.LEFT, padx=5)

        # Visualization area
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Airspace Visualization")
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Please load an airspace", 
                    ha='center', va='center', fontsize=12, color='gray')

        self.canvas = FigureCanvasTkAgg(self.figure, master=vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, vis_frame)
        self.toolbar.update()

    def load_airspace(self, region):
        base_dir = "data"
        try:
            nav_file = os.path.join(base_dir, f"{region[:3]}_nav.txt")
            seg_file = os.path.join(base_dir, f"{region[:3]}_seg.txt")
            aer_file = os.path.join(base_dir, f"{region[:3]}_aer.txt")
            
            self.current_airspace = AirSpace(region)
            self.current_airspace.load_from_files(nav_file, seg_file, aer_file)
            self.convert_to_graph()
            self.refresh_visualization()
            messagebox.showinfo("Success", f"{region} airspace loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {region}: {str(e)}")
            print(f"Error details: {str(e)}")

    def convert_to_graph(self):
        """Convert AirSpace to Graph structure"""
        from Graph import Graph, Node, Segment
        
        self.graph = Graph()
        
        # Add all navpoints as nodes
        for number, navpoint in self.current_airspace.navpoints.items():
            node = Node(
                name=navpoint.name,
                x=navpoint.longitude,
                y=navpoint.latitude,
                number=number
            )
            node.is_airport = navpoint.is_airport
            self.graph.nodes[navpoint.name] = node
        
        # Add all segments
        for navsegment in self.current_airspace.navsegments:
            origin = self.current_airspace.get_navpoint(navsegment.origin_number)
            dest = self.current_airspace.get_navpoint(navsegment.destination_number)
            
            if origin and dest and origin.name in self.graph.nodes and dest.name in self.graph.nodes:
                segment_name = f"{origin.name}-{dest.name}"
                segment = Segment(
                    name=segment_name,
                    origin=self.graph.nodes[origin.name],
                    destination=self.graph.nodes[dest.name],
                    cost=navsegment.distance
                )
                self.graph.segments[segment_name] = segment
                self.graph.nodes[origin.name].neighbors.append(self.graph.nodes[dest.name])

    def refresh_visualization(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        if not self.graph:
            self.ax.text(0.5, 0.5, "Please load an airspace", 
                        ha='center', va='center', fontsize=12, color='gray')
            self.ax.axis('off')
            self.canvas.draw()
            return
        
        # Plot all segments
        for segment in self.graph.segments.values():
            self.ax.plot(
                [segment.origin.x, segment.destination.x],
                [segment.origin.y, segment.destination.y],
                'b-', alpha=0.3, linewidth=0.7
            )
            
            # Add distance labels
            mid_x = (segment.origin.x + segment.destination.x) / 2
            mid_y = (segment.origin.y + segment.destination.y) / 2
            self.ax.text(mid_x, mid_y, f"{segment.cost:.1f}", 
                        color='red', fontsize=8, backgroundcolor=(1, 1, 1, 0.7))
        
        # Plot all nodes
        for node in self.graph.nodes.values():
            if hasattr(node, 'is_airport') and node.is_airport:
                # Airport nodes (green squares)
                self.ax.plot(node.x, node.y, 's', color='green', markersize=10)
                self.ax.text(node.x, node.y, f" {node.name}", 
                            color='green', fontsize=10, weight='bold')
            else:
                # Regular navigation points (blue circles)
                self.ax.plot(node.x, node.y, 'o', color='blue', markersize=5)
                self.ax.text(node.x, node.y, f" {node.name}", 
                            color='blue', fontsize=8)
        
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title(f"{self.current_airspace.region} Airspace")
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def show_neighbors(self):
        if not self.graph:
            messagebox.showerror("Error", "Please load an airspace first")
            return
            
        node_name = self.node_entry.get().strip()
        if not node_name:
            messagebox.showerror("Error", "Please enter a node name")
            return
            
        if node_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Node '{node_name}' not found")
            return
            
        node = self.graph.nodes[node_name]
        neighbors = node.neighbors
        
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Plot base graph
        self.refresh_visualization()
        
        # Highlight selected node
        self.ax.plot(node.x, node.y, 'ro', markersize=10)
        self.ax.text(node.x, node.y, f" {node.name}", 
                    color='red', fontsize=10, weight='bold')
        
        # Highlight neighbors
        for neighbor in neighbors:
            self.ax.plot(neighbor.x, neighbor.y, 'yo', markersize=8)
            self.ax.text(neighbor.x, neighbor.y, f" {neighbor.name}", 
                        color='orange', fontsize=9)
            
            # Highlight connecting segments
            seg_name = f"{node.name}-{neighbor.name}"
            if seg_name in self.graph.segments:
                self.ax.plot(
                    [node.x, neighbor.x],
                    [node.y, neighbor.y],
                    'r-', linewidth=1.5
                )
        
        self.canvas.draw()
        messagebox.showinfo("Neighbors", 
                          f"Node {node_name} has {len(neighbors)} neighbors:\n" +
                          "\n".join([n.name for n in neighbors]))

    def show_reachable_nodes(self):
        if not self.graph:
            messagebox.showerror("Error", "Please load an airspace first")
            return
            
        node_name = self.node_entry.get().strip()
        if not node_name:
            messagebox.showerror("Error", "Please enter a node name")
            return
            
        if node_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Node '{node_name}' not found")
            return
            
        from Graph import FindReachableNodes
        reachable_nodes = FindReachableNodes(self.graph, node_name)
        
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Plot base graph
        self.refresh_visualization()
        
        # Highlight reachable nodes
        for node in reachable_nodes:
            self.ax.plot(node.x, node.y, 'go', markersize=8)
            self.ax.text(node.x, node.y, f" {node.name}", 
                        color='green', fontsize=9)
        
        # Highlight starting node
        start_node = self.graph.nodes[node_name]
        self.ax.plot(start_node.x, start_node.y, 'ro', markersize=10)
        self.ax.text(start_node.x, start_node.y, f" {start_node.name}", 
                    color='red', fontsize=10, weight='bold')
        
        self.canvas.draw()
        messagebox.showinfo("Reachable Nodes", 
                          f"Found {len(reachable_nodes)} reachable nodes from {node_name}")

    def find_shortest_path(self):
        if not self.graph:
            messagebox.showerror("Error", "Please load an airspace first")
            return
            
        origin_name = self.origin_entry.get().strip()
        dest_name = self.dest_entry.get().strip()
        
        if not origin_name or not dest_name:
            messagebox.showerror("Error", "Please enter both origin and destination")
            return
            
        if origin_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Origin node '{origin_name}' not found")
            return
            
        if dest_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Destination node '{dest_name}' not found")
            return
            
        from Graph import FindShortestPath, PlotPath
        path = FindShortestPath(self.graph, origin_name, dest_name)
        
        if not path:
            messagebox.showinfo("Path", f"No path found between {origin_name} and {dest_name}")
            return
            
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Plot base graph
        self.refresh_visualization()
        
        # Highlight the path
        PlotPath(self.graph, path, self.ax)
        
        # Show path info
        path_names = " → ".join([node.name for node in path.nodes])
        messagebox.showinfo("Shortest Path",
                          f"Path: {path_names}\n"
                          f"Total distance: {path.cost:.1f} km")
        
        self.canvas.draw()


if __name__ == "__main__":
    app = GraphInterface()
    app.mainloop()
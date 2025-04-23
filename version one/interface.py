import tkinter as tk
from tkinter import ttk, messagebox
from path import PlotPath 

class GraphInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizador de Grafos Avanzado")
        self.geometry("1100x850")
        self.graph = None
        
        # Configurar el estilo de ttk
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Tema con buen contraste (opciones: 'clam', 'alt', 'default', 'classic')
        
        # Crear interfaz
        self.create_widgets()
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from Graph import FindReachableNodes, FindShortestPath, Graph, Node, AddNode, AddSegment, ReadGraphFromFile, Plot, PlotNode, GetClosest

class GraphInterface(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Visualizador de Grafos Avanzado")
        self.geometry("1100x850")
        self.graph = None
        
        # Configurar tema ttk
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Tema con buen contraste
        
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
            #UPDATE VERSION 2
            ("Nodos Alcanzables", self.show_reachable_nodes),
            ("Camino más Corto (A*)", self.show_shortest_path)]
        
        for text, command in edit_buttons:
            btn = ttk.Button(
                edit_frame, 
                text=text, 
                command=command,
                style="TButton")
            btn.pack(side=tk.LEFT, padx=5)
        
        # Área del gráfico (se mantiene tk.Frame para Matplotlib)
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
            Plot(self.graph, self.ax)  # Pasar self.ax explícitamente
        else:
            self.ax.text(0.5, 0.5, "Seleccione una opción para visualizar un grafo", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.axis("off")
        self.canvas.draw()
    
    def load_example_graph(self):
        try:
            # Importación absoluta desde el mismo directorio
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
        """Carga un grafo desde archivo"""
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
                    # Escribir nodos
                    f.write("NODES\n")
                    for node in self.graph.nodes.values():  # Acceso a valores del diccionario
                        f.write(f"{node.name} {node.x} {node.y}\n")
                    
                    # Escribir segmentos
                    f.write("\nSEGMENTS\n")
                    for segment in self.graph.segments.values():  # Acceso a valores del diccionario
                        f.write(f"{segment.name} {segment.origin.name} {segment.destination.name} {segment.cost}\n")
                
                messagebox.showinfo("Éxito", f"Grafo guardado correctamente en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el grafo:\n{str(e)}")

    def show_node_neighbors(self):
        """Versión corregida para mostrar nodos"""
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
            
            # Dibujar el grafo completo
            Plot(self.graph, self.ax)
            
            # Resaltar el nodo específico
            if not PlotNode(self.graph, node_name, self.ax):
                messagebox.showerror("Error", f"Nodo '{node_name}' no encontrado")
                self.refresh_graph()
                return
                
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar nodo:\n{str(e)}")
            self.refresh_graph()
    
    def add_node(self):
        """Añade un nodo al grafo (Step 6)"""
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
        """Añade un segmento al grafo (Step 6)"""
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
            # Eliminar segmentos relacionados (ahora es un diccionario)
            segments_to_remove = [
                seg_name for seg_name, seg in self.graph.segments.items()
                if seg.origin.name == node_name or seg.destination.name == node_name]
            
            for seg_name in segments_to_remove:
                del self.graph.segments[seg_name]
            
            # Eliminar el nodo
            del self.graph.nodes[node_name]
            
            self.refresh_graph()
            messagebox.showinfo("Éxito", f"Nodo '{node_name}' y {len(segments_to_remove)} segmentos eliminados")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el nodo:\n{str(e)}")
    
    def find_closest_node(self):
        """Encuentra el nodo más cercano a unas coordenadas (Step 6)"""
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
                
                # Resaltar el nodo en el gráfico
                self.figure.clear()
                self.ax = self.figure.add_subplot(111)
                Plot(self.graph, self.ax)
                PlotNode(self.graph, closest_node.name, self.ax)
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo encontrar el nodo más cercano:\n{str(e)}")
if __name__ == "__main__":
    app = GraphInterface()
    app.mainloop()

#HERE ADDITIONS V2
def show_reachable_nodes(self):
    """Show all nodes reachable from a starting node"""
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
        
        # Draw the graph
        Plot(self.graph, self.ax)
        
        # Highlight reachable nodes
        for node in reachable:
            self.ax.plot(node.x, node.y, 'go', markersize=10)
            self.ax.text(node.x, node.y, node.name, fontsize=12, ha='right')
        
        # Highlight starting node
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
    """Show shortest path between two nodes using A* algorithm"""
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
        
        # Draw the graph
        Plot(self.graph, self.ax)
        
        # Highlight the path
        PlotPath(self.graph, path, self.ax)
        
        # Show info
        messagebox.showinfo("Camino más corto", 
                          f"Camino: {path}\n"
                          f"Costo total: {path.cost:.2f}")
        
        self.canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo calcular el camino:\n{str(e)}")
        self.refresh_graph()
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from Graph import Graph, Node, AddNode, AddSegment, ReadGraphFromFile, Plot, PlotNode, GetClosest

class GraphInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Graph Visualizer")
        self.geometry("1100x850")
        self.graph = None
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#4a7a8c"
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TButton", 
                           background=self.accent_color, 
                           foreground="white",
                           font=("Helvetica", 10, "bold"),
                           padding=5)
        
        # Algorithm state variables
        self.current_mode = None
        self.path_start = None
        self.path_end = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all UI components"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Control Panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Title
        title_label = ttk.Label(control_frame, 
                              text="GRAPH VISUALIZATION TOOL", 
                              font=("Helvetica", 20, "bold"))
        title_label.pack(pady=10)
        
        # Main buttons - REMOVED load_custom_graph since it wasn't implemented
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        main_buttons = [
            ("Example Graph", self.load_example_graph),
            ("Load from File", self.load_from_file),
            ("Save Graph", self.save_graph)
        ]
        
        for text, cmd in main_buttons:
            btn = ttk.Button(button_frame, text=text, command=cmd)
            btn.pack(side=tk.LEFT, padx=5)
        
        # Search and Edit Panel
        edit_frame = ttk.Frame(control_frame)
        edit_frame.pack(pady=10)
        
        # Search
        ttk.Label(edit_frame, text="Search Node:").pack(side=tk.LEFT, padx=5)
        self.node_entry = ttk.Entry(edit_frame, width=20)
        self.node_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(edit_frame, 
                  text="Show Neighbors", 
                  command=self.show_node_neighbors).pack(side=tk.LEFT, padx=5)
        
        # Edit buttons
        edit_buttons = [
            ("Add Node", self.add_node),
            ("Add Edge", self.add_segment),
            ("Delete Node", self.delete_node),
            ("Find Closest", self.find_closest_node)
        ]
        
        for text, cmd in edit_buttons:
            ttk.Button(edit_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=5)
        
        # Algorithm Controls
        algo_frame = ttk.LabelFrame(control_frame, text="Algorithms", padding=10)
        algo_frame.pack(fill=tk.X, pady=10)
        
        algo_buttons = [
            ("Show Reachable", self.prepare_reachability),
            ("Set Start Node", lambda: self.set_path_mode("start")),
            ("Set End Node", lambda: self.set_path_mode("end")),
            ("Find Shortest Path", self.find_shortest_path)
        ]
        
        for text, cmd in algo_buttons:
            ttk.Button(algo_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=5)
        
        # Status Bar
        self.status_var = tk.StringVar()
        ttk.Label(control_frame, 
                 textvariable=self.status_var, 
                 relief=tk.SUNKEN).pack(fill=tk.X, pady=5)
        
        # Graph Display
        graph_frame = tk.Frame(main_frame, bg="white")
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, "Load or create a graph to begin", 
                    ha="center", va="center", fontsize=12, color="gray")
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        self.toolbar.update()
        
    def refresh_graph(self, highlight_nodes=None, highlight_edges=None):
        """Redraw the graph with optional highlights"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        if not self.graph:
            self.ax.text(0.5, 0.5, "Load or create a graph to begin", 
                        ha="center", va="center", fontsize=12, color="gray")
            self.ax.axis("off")
        else:
            # Draw all nodes
            for node in self.graph.nodes.values():
                color = "green" if highlight_nodes and node.name in highlight_nodes else "lightblue"
                self.ax.plot(node.x, node.y, 'o', markersize=15, color=color)
                self.ax.text(node.x, node.y, node.name, ha='center', va='center')
            
            # Draw all edges
            for edge in self.graph.segments.values():
                color = "orange" if highlight_edges and edge.name in highlight_edges else "gray"
                x1, y1 = edge.origin.x, edge.origin.y
                x2, y2 = edge.destination.x, edge.destination.y
                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)
        
        self.canvas.draw()
    
    def load_example_graph(self):
            
        try:
            from test_graphy2 import CreateGraph_1
            self.graph = CreateGraph_1()
        
            # Clear and prepare the plot
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
        
            # Plot all nodes with their names
            for node in self.graph.nodes.values():
                self.ax.plot(node.x, node.y, 'o', markersize=12, 
                        color='lightblue', markeredgecolor='black')
                self.ax.text(node.x, node.y+0.5, node.name,  # Offset label above node
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
            # Plot all edges with distance labels
            for edge in self.graph.segments.values():
                x1, y1 = edge.origin.x, edge.origin.y
                x2, y2 = edge.destination.x, edge.destination.y
            
            # Calculate Euclidean distance
                distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            
            # Draw the edge line
                self.ax.plot([x1, x2], [y1, y2], color='gray', linewidth=2, zorder=1)
            
            # Add distance label at midpoint (offset slightly)
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                angle = 45  # Angle for better label placement
                self.ax.text(mid_x, mid_y, f"{distance:.2f}", 
                        color='darkred', fontsize=9, ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='none', pad=1),
                        rotation=angle, zorder=2)
        
        # Set plot limits with padding
            all_x = [node.x for node in self.graph.nodes.values()]
            all_y = [node.y for node in self.graph.nodes.values()]
            padding = 2
            self.ax.set_xlim(min(all_x)-padding, max(all_x)+padding)
            self.ax.set_ylim(min(all_y)-padding, max(all_y)+padding)
        
        # Configure plot appearance
            self.ax.set_title("Air Routes Graph with Distances", pad=20)
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.ax.set_aspect('equal')
        
            self.canvas.draw()
            self.status_var.set("Example graph loaded successfully")
        
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't load example graph:\n{str(e)}")
        # Reset the plot on error
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, "Error loading graph", 
                    ha='center', va='center', color='red')
            self.ax.axis('off')
            self.canvas.draw()
            
    def load_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                self.graph = ReadGraphFromFile(file_path)
                self.status_var.set(f"Graph loaded: {len(self.graph.nodes)} nodes, {len(self.graph.segments)} edges")
                self.refresh_graph()
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't load file:\n{str(e)}")
    
    def save_graph(self):
        if not self.graph:
            messagebox.showwarning("Warning", "No graph to save")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("NODES\n")
                    for node in self.graph.nodes.values():
                        f.write(f"{node.name} {node.x} {node.y}\n")
                    
                    f.write("\nEDGES\n")
                    for edge in self.graph.segments.values():
                        f.write(f"{edge.name} {edge.origin.name} {edge.destination.name} {edge.cost}\n")
                
                self.status_var.set(f"Graph saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't save file:\n{str(e)}")
    
    def add_node(self):
        name = simpledialog.askstring("Add Node", "Node name:")
        if name:
            try:
                x = float(simpledialog.askstring("Coordinates", "X position:"))
                y = float(simpledialog.askstring("Coordinates", "Y position:"))
                AddNode(self.graph, Node(name, x, y))
                self.status_var.set(f"Added node {name}")
                self.refresh_graph()
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't add node:\n{str(e)}")
    
    def add_segment(self):
        origin = simpledialog.askstring("Add Edge", "Origin node:")
        dest = simpledialog.askstring("Add Edge", "Destination node:")
        if origin and dest:
            try:
                edge_name = f"{origin}-{dest}"
                AddSegment(self.graph, edge_name, origin, dest)
                self.status_var.set(f"Added edge {edge_name}")
                self.refresh_graph()
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't add edge:\n{str(e)}")
    
    def delete_node(self):
        node_name = simpledialog.askstring("Delete Node", "Node name to delete:")
        if node_name and node_name in self.graph.nodes:
            try:
                # Remove connected edges first
                edges_to_remove = [name for name, edge in self.graph.segments.items() 
                                 if edge.origin.name == node_name or edge.destination.name == node_name]
                for edge_name in edges_to_remove:
                    del self.graph.segments[edge_name]
                
                # Remove the node
                del self.graph.nodes[node_name]
                self.status_var.set(f"Deleted node {node_name} and {len(edges_to_remove)} edges")
                self.refresh_graph()
            except Exception as e:
                messagebox.showerror("Error", f"Couldn't delete node:\n{str(e)}")
    
    def find_closest_node(self):
        try:
            x = float(simpledialog.askstring("Coordinates", "X position:"))
            y = float(simpledialog.askstring("Coordinates", "Y position:"))
            closest = GetClosest(self.graph, x, y)
            if closest:
                self.refresh_graph(highlight_nodes=[closest.name])
                self.status_var.set(f"Closest node: {closest.name} at ({closest.x}, {closest.y})")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't find closest node:\n{str(e)}")
    
    def show_node_neighbors(self):
        node_name = self.node_entry.get()
        if node_name in self.graph.nodes:
            neighbors = set()
            for edge in self.graph.segments.values():
                if edge.origin.name == node_name:
                    neighbors.add(edge.destination.name)
                elif edge.destination.name == node_name:
                    neighbors.add(edge.origin.name)
            
            if neighbors:
                self.refresh_graph(highlight_nodes=list(neighbors))
                self.status_var.set(f"Neighbors of {node_name}: {', '.join(neighbors)}")
            else:
                messagebox.showinfo("Info", f"Node {node_name} has no neighbors")
        else:
            messagebox.showerror("Error", f"Node {node_name} not found")
    
    def prepare_reachability(self):
        self.current_mode = "reachability"
        self.status_var.set("Click a node to show reachable nodes")
    
    def set_path_mode(self, node_type):
        self.current_mode = f"path_{node_type}"
        self.status_var.set(f"Click a node to set {node_type} point")
    
    def handle_node_click(self, node_name):
        """Handle node clicks from matplotlib plot"""
        if not self.graph or node_name not in self.graph.nodes:
            return
            
        if self.current_mode == "reachability":
            try:
                reachable = self.graph.find_reachable_nodes(node_name)
                self.refresh_graph(highlight_nodes=reachable)
                self.status_var.set(f"Found {len(reachable)} reachable nodes")
            except Exception as e:
                messagebox.showerror("Error", f"Reachability failed:\n{str(e)}")
        elif self.current_mode == "path_start":
            self.path_start = node_name
            self.status_var.set(f"Start node set to {node_name}")
        elif self.current_mode == "path_end":
            self.path_end = node_name
            self.status_var.set(f"End node set to {node_name}")
        
        self.current_mode = None
    
    def find_shortest_path(self):
        if not self.path_start or not self.path_end:
            messagebox.showwarning("Warning", "Please select both start and end nodes")
            return
            
        try:
            path = self.graph.find_shortest_path(self.path_start, self.path_end)
            if path:
                highlight_edges = [f"{path[i]}-{path[i+1]}" for i in range(len(path)-1)]
                self.refresh_graph(highlight_nodes=path, highlight_edges=highlight_edges)
                self.status_var.set(f"Shortest path found (cost: {self.graph.path_cost(path):.2f})")
            else:
                messagebox.showinfo("No Path", "No path exists between selected nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Path finding failed:\n{str(e)}")

if __name__ == "__main__":
    app = GraphInterface()
    app.mainloop()
# interface.py (Step 5 with embedded plotting)
# interface.py (Step 5 with embedded plotting, fixed backend)
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg before any other Matplotlib imports

import tkinter as tk
from tkinter import filedialog, messagebox
from Graph import Graph, Node, AddNode, AddSegment, ReadGraphFromFile
from test_graphy import CreateGraph_1, CreateGraph_2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

print("Starting interface.py")

class GraphInterface:
    def __init__(self, root):
        print("Initializing GraphInterface...")
        self.root = root
        self.root.title("Graph Interface (Step 5)")
        self.graph = Graph()  # Initialize an empty graph
        self.fig = None
        self.canvas = None
        self.create_widgets()
        print("GraphInterface initialized")

    def create_widgets(self):
        print("Creating widgets...")
        # Frame for buttons and inputs
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(button_frame, text="Show Graph 1 (Step 3)", command=self.show_graph1).pack(pady=5)
        tk.Button(button_frame, text="Show Graph 2 (Step 4)", command=self.show_graph2).pack(pady=5)
        tk.Button(button_frame, text="Load Graph from File", command=self.load_graph).pack(pady=5)
        tk.Label(button_frame, text="Show Node Neighbors").pack(pady=5)
        self.node_entry = tk.Entry(button_frame)
        self.node_entry.pack()
        tk.Button(button_frame, text="Show Neighbors", command=self.show_node_neighbors).pack(pady=5)
        tk.Label(button_frame, text="Add Node (name x y)").pack(pady=5)
        self.add_node_entry = tk.Entry(button_frame)
        self.add_node_entry.pack()
        tk.Button(button_frame, text="Add Node", command=self.add_node).pack(pady=5)
        tk.Label(button_frame, text="Add Segment (origin destination cost)").pack(pady=5)
        self.add_segment_entry = tk.Entry(button_frame)
        self.add_segment_entry.pack()
        tk.Button(button_frame, text="Add Segment", command=self.add_segment).pack(pady=5)

        # Frame for plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        print("Widgets created")

    def clear_plot(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.canvas = None

    def plot_graph(self, plot_func, *args):
        self.clear_plot()
        self.fig = plt.Figure(figsize=(8, 6))
        ax = self.fig.add_subplot(111)

        # Call the plotting function
        plot_func(*args, ax=ax)

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_full_graph(self, g, ax):
        # Draw segments
        for segment in g.segments:
            ax.plot([segment.origin.x, segment.destination.x],
                    [segment.origin.y, segment.destination.y],
                    'k-', alpha=0.5)
            mid_x = (segment.origin.x + segment.destination.x) / 2
            mid_y = (segment.origin.y + segment.destination.y) / 2
            ax.text(mid_x, mid_y, str(segment.cost), color='red')

        # Draw nodes
        for node in g.nodes:
            ax.plot(node.x, node.y, 'bo')
            ax.text(node.x, node.y, node.name, fontsize=12, ha='right')

        ax.grid(True)

    def plot_node(self, g, nameOrigin, ax):
        origin = g.find_node(nameOrigin)
        if origin is None:
            return False

        # Draw all segments first (gray)
        for segment in g.segments:
            ax.plot([segment.origin.x, segment.destination.x],
                    [segment.origin.y, segment.destination.y],
                    'k-', color='gray', alpha=0.3)

        # Highlight origin node and its connections
        for segment in g.segments:
            if segment.origin == origin or segment.destination == origin:
                ax.plot([segment.origin.x, segment.destination.x],
                        [segment.origin.y, segment.destination.y],
                        'r-', linewidth=2)
                mid_x = (segment.origin.x + segment.destination.x) / 2
                mid_y = (segment.origin.y + segment.destination.y) / 2
                ax.text(mid_x, mid_y, str(segment.cost), color='red')

        # Draw all nodes
        for node in g.nodes:
            if node == origin:
                ax.plot(node.x, node.y, 'bo', markersize=10)
            elif node in origin.neighbors:
                ax.plot(node.x, node.y, 'go', markersize=8)
            else:
                ax.plot(node.x, node.y, 'o', color='gray', markersize=6)

            ax.text(node.x, node.y, node.name, fontsize=12, ha='right')

        ax.grid(True)
        return True

    def show_graph1(self):
        print("Showing Graph 1 (Step 3)...")
        self.graph = CreateGraph_1()
        self.plot_graph(self.plot_full_graph, self.graph)

    def show_graph2(self):
        print("Showing Graph 2 (Step 4)...")
        self.graph = CreateGraph_2()
        self.plot_graph(self.plot_full_graph, self.graph)

    def load_graph(self):
        print("Loading graph from file...")
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            print("No file selected")
            return
        self.graph = ReadGraphFromFile(file_path)
        if self.graph is not None:
            print(f"Graph loaded with {len(self.graph.nodes)} nodes and {len(self.graph.segments)} segments.")
            self.plot_graph(self.plot_full_graph, self.graph)
        else:
            messagebox.showerror("Error", "Failed to load graph from file")

    def show_node_neighbors(self):
        node_name = self.node_entry.get().strip()
        print(f"Showing neighbors for node {node_name}...")
        if not node_name:
            messagebox.showerror("Error", "Please enter a node name")
            return
        success = self.plot_graph(self.plot_node, self.graph, node_name)
        if not success:
            messagebox.showerror("Error", f"Node {node_name} not found")

    def add_node(self):
        try:
            name, x, y = self.add_node_entry.get().strip().split()
            print(f"Adding node {name} at ({x}, {y})...")
            new_node = Node(name, float(x), float(y))
            if AddNode(self.graph, new_node):
                messagebox.showinfo("Success", f"Node {name} added")
                self.plot_graph(self.plot_full_graph, self.graph)
            else:
                messagebox.showerror("Error", f"Node {name} already exists")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}\nFormat: name x y")

    def add_segment(self):
        try:
            parts = self.add_segment_entry.get().strip().split()
            if len(parts) < 2:
                raise ValueError("At least origin and destination are required")
            origin, dest = parts[0], parts[1]
            cost = float(parts[2]) if len(parts) > 2 else 1.0
            seg_name = f"{origin}-{dest}"
            print(f"Adding segment {seg_name} with cost {cost}...")
            if AddSegment(self.graph, seg_name, origin, dest, cost):
                messagebox.showinfo("Success", f"Segment {seg_name} added")
                self.plot_graph(self.plot_full_graph, self.graph)
            else:
                messagebox.showerror("Error", "One or both nodes not found")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}\nFormat: origin destination [cost]")
            
            
    print("Creating Tkinter window...")
    root = tk.Tk()
    print("Tkinter window created")
    app = GraphInterface(root)
    print("Starting main loop...")
    root.mainloop()
    print("Main loop ended")
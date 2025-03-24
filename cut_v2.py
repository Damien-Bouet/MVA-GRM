import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion
import maxflow
from tqdm import tqdm
import tracemalloc
import gc
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from time import time
import networkx

import os
class SeedLabeler:
    def __init__(self, image, image_path=None, no_fig=False):
        self.image = image
        self.image_path = image_path  # Store image path for reloading
        self.foreground = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        self.background = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        self.drawing = False  # Track whether mouse button is held
        self.current_label = None  # Track foreground or background labeling
        self.previous_pos = None
        
        if not no_fig:
            self.fig, self.ax = plt.subplots()
            self.ax.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
            self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def on_press(self, event):
        if event.inaxes != self.ax: return
        x, y = int(event.xdata), int(event.ydata)
        self.drawing = True
        if event.button == 1:
            self.current_label = 'foreground'
        elif event.button == 3:
            self.current_label = 'background'
        self.draw_circle(x, y)
    
    def on_release(self, event):
        self.previous_pos = None
        self.drawing = False
        self.current_label = None
    
    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            self.draw_circle(x, y)
    
    def draw_circle(self, x, y):
        if self.previous_pos is not None:
            n = max(abs(self.previous_pos[0] - x), abs(self.previous_pos[1] - y))
            for x_, y_ in zip(np.linspace(self.previous_pos[0], x, n), np.linspace(self.previous_pos[1], y, n)):
                if self.current_label == 'foreground':
                    self.foreground[int(y_):int(y_)+2, int(x_):int(x_)+2] = True
                elif self.current_label == 'background':
                    self.background[int(y_):int(y_)+2, int(x_):int(x_)+2] = True
                    
        if self.current_label == 'foreground':
            self.foreground[y:y+2, x:x+2] = True
        elif self.current_label == 'background':
            self.background[y:y+2, x:x+2] = True
            
        self.previous_pos = (x, y)
        self.update_display()
    
    def update_display(self):
        disp = self.image.copy()
        if len(self.image.shape) == 2:
            disp = np.concatenate([disp[:, :, None]]*3, axis=2)
        
        disp[self.foreground.astype(bool)] = [1, 1, 0]  # Yellow
        disp[self.background.astype(bool)] = [0, 0, 1]  # Blue
        self.ax.imshow(disp)
        self.fig.canvas.draw()
    
    def save_seeds(self, save_dir):
        """Save seeds to numpy files"""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'foreground.npy'), self.foreground)
        np.save(os.path.join(save_dir, 'background.npy'), self.background)
        if self.image_path:
            with open(os.path.join(save_dir, 'image_path.txt'), 'w') as f:
                f.write(self.image_path)
    
    @classmethod
    def load_seeds(cls, save_dir):
        """Load seeds from directory"""
        image_path = None
        if os.path.exists(os.path.join(save_dir, 'image_path.txt')):
            with open(os.path.join(save_dir, 'image_path.txt'), 'r') as f:
                image_path = f.read().strip()
        
        if image_path and os.path.exists(image_path):
            image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
        else:
            raise FileNotFoundError("Original image not found")
        
        labeler = cls(image, image_path, no_fig=True)
        labeler.foreground = np.load(os.path.join(save_dir, 'foreground.npy')).astype(bool)
        labeler.background = np.load(os.path.join(save_dir, 'background.npy')).astype(bool)
        return labeler

def coarsen(img, factor=2):
    """Coarsen image with padding if needed"""
    h, w = img.shape[:2]
    # Pad if dimensions aren't divisible by factor
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_AREA)

def coarsen_seeds(seeds, factor=2):
    """Coarsen seeds with padding to handle non-divisible dimensions"""
    h, w = seeds.shape
    # Pad to make dimensions divisible by factor
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    padded = np.pad(seeds, ((0, pad_h), (0, pad_w)), mode='constant')

    # Reshape and take max over blocks
    h_p, w_p = padded.shape
    coarse = padded.reshape(h_p//factor, factor, w_p//factor, factor).max(axis=(1, 3)).astype(bool)
    return coarse[:h//factor, :w//factor]


# ----------------------------
# 1. Band-Limited Graph Construction
# ----------------------------

def create_memory_optimized_graph(img, band_mask, prev_seg, sigma):
    """Memory-efficient graph construction"""
    h, w = img.shape[:2]
    graph = maxflow.GraphFloat()
    
    # 1. Add nodes in bulk for band pixels
    band_pixels = np.argwhere(band_mask)
    node_ids = graph.add_nodes(len(band_pixels))
    node_map = {(y,x): idx for idx, (y,x) in enumerate(band_pixels)}
    
    # 2. Process in chunks to reduce peak memory
    chunk_size = 10000
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(0, len(band_pixels), chunk_size):
        chunk = band_pixels[i:i+chunk_size]
        
        # Add t-links
        for y, x in chunk:
            if prev_seg[y,x]:
                graph.add_tedge(node_map[(y,x)], 0, 1)
            else:
                graph.add_tedge(node_map[(y,x)], 1, 0)
        
        # Add n-links
        for y, x in chunk:
            for dy, dx in offsets:
                ny, nx = y+dy, x+dx
                if (ny,nx) in node_map:
                    diff = np.sum((img[y,x]-img[ny,nx])**2)
                    weight = np.exp(-diff/(2*sigma**2))
                    graph.add_edge(node_map[(y,x)], node_map[(ny,nx)], weight, weight)
    
    return graph, node_map

# ----------------------------
# 2. Multilevel Banded Cuts
# ----------------------------

def memory_optimized_banded_cuts(image, fg_seeds, bg_seeds, levels=3, band_width=2, factor = 2, sigma=0.1):
    """Memory-optimized version with proper visualization"""
    # Initialize results storage
    all_results = []
    #0. Baseline, regular GC
    print("Computing baseline GC...")
    gc.collect()
    tracemalloc.start()
    t0 = time()
    regular_seg = regular_graph_cuts(image,fg_seeds, bg_seeds, sigma)
    t1 = time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    all_results.append({
        'type': 'baseline',
        'time': t1 - t0,
        'memory': peak_mem / 1e6,
        'image': image.copy(),
        'segmentation': regular_seg.copy()
    })

    # 1. Coarsening stage with memory tracking
    current_img, current_fg, current_bg = image, fg_seeds, bg_seeds

    all_results.append({
        'type': 'original',
        'image': image.copy(),
        'fg_seeds': fg_seeds,
        "bg_seeds": bg_seeds
        
    })

    pyramids = [(image, fg_seeds, bg_seeds)]
    for level in range(levels):
        # Create pyramid level
    
        new_img = coarsen(pyramids[-1][0], factor)
        new_fg = coarsen_seeds(pyramids[-1][1], factor)
        new_bg = coarsen_seeds(pyramids[-1][2], factor)
        pyramids.append((new_img, new_fg, new_bg))
        
        # Clean up
        del current_img, current_fg, current_bg
        current_img, current_fg, current_bg = new_img, new_fg, new_bg
    # 2. Process coarsest level
    gc.collect()
    tracemalloc.start()
    t0 = time()
    seg = regular_graph_cuts(current_img, current_fg, current_bg, sigma)
    t1 = time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    all_results.append({
        'level': levels-1,
        'type': 'segmentation',
        'time': t1 - t0,
        'memory': peak_mem / 1e6,
        'image': current_img.copy(),
        'segmentation': seg.copy()
    })

    
    # 3. Uncoarsening with band refinement
    for level in reversed(range(levels-1)):
        current_seg = seg.copy()
        print(current_seg.shape)
        
        t0 = time()
        
        # Project segmentation
        curr_img = pyramids[level][0]
        seg = cv2.resize(current_seg.copy().astype(float), curr_img.shape[:2][::-1], 
                        interpolation=cv2.INTER_NEAREST) > 0.5
        
        print(current_seg.shape, curr_img.shape)
        
        # Create band
        band_mask = (binary_dilation(seg, iterations=band_width) ^ 
                    binary_erosion(seg, iterations=band_width))
        
        # Build and solve banded graph
        gc.collect()
        tracemalloc.start()
        graph, node_map = create_memory_optimized_graph(curr_img, band_mask, seg, sigma)
        graph.maxflow()
        
        # Update segmentation

        for (y,x), nodeid in node_map.items():
            seg[y,x] = bool(graph.get_segment(nodeid))
        
        t1 = time()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        all_results.append({
            'level': level,
            'type': 'refinement',
            'time': t1 - t0,
            'memory': peak_mem / 1e6,
            'image': curr_img.copy(),
            'segmentation': seg,
            'band_mask': band_mask.copy()
        })



    # 4. Create memory-efficient visualization
    create_detailed_visualization(all_results, image.shape[1]/image.shape[0])
    
    return seg

def visualize_node_map(node_map, img_shape):
    """Visualize the assigned node IDs in the image space"""
    vis = np.full(img_shape[:2], -1, dtype=int)  # Fill with -1 for non-band pixels
    for (y, x), node_id in node_map.items():
        vis[y, x] = node_id
    
    plt.figure(figsize=(8, 6))
    plt.imshow(vis, cmap='jet', interpolation='nearest')
    plt.colorbar(label="Node ID")
    plt.title("Node Map Visualization")
    plt.show()



def visualize_graph_edges(node_map):
    """Visualize graph structure using NetworkX"""
    G = networkx.Graph()
    
    for (y, x), node_id in node_map.items():
        G.add_node(node_id, pos=(x, -y))  # Flip y-axis for visualization
        
        # Check neighbors
        for dy, dx in [(0,1), (1,0)]:  # Only right and down (avoid duplicates)
            ny, nx = y+dy, x+dx
            if (ny, nx) in node_map:
                G.add_edge(node_id, node_map[(ny, nx)])

    # Draw the graph
    pos = networkx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 10))
    networkx.draw(G, pos, node_size=10, edge_color='gray', alpha=0.5, with_labels=False)
    plt.title("Graph Structure (Edges & Nodes)")

def create_detailed_visualization(results, aspect_ratio):
    """Enhanced visualization showing segmentation at each step"""
    # Filter and organize results
    seg_steps = [r for r in results if r['type'] in ['segmentation', 'refinement']]
    
    # Create figure layout
    n_cols = len(seg_steps) + 1
    fig, axs = plt.subplots(2, n_cols, 
                          figsize=(n_cols*4*aspect_ratio, 8),
                          constrained_layout=True)
    
    #Plot baseline
    baseline = results[0]
    seg_vis = np.zeros((*baseline['segmentation'].shape, 4))
    seg_vis[~baseline['segmentation']] = [1, 1, 0, 0.6]  # Yellow FG
    seg_vis[baseline['segmentation']] = [0, 0, 1, 0.6]   # Blue BG
    axs[0,2].imshow(baseline['image'], cmap='gray' if len(baseline['image'].shape) == 2 else None)
    axs[0,2].imshow(seg_vis)
    axs[0,2].set_title(f"Baseline\n(Time spent: {baseline['time']:.2g}s | Memory: {baseline['memory']:.2g}MB")
    # axs[0,1].axis('off') 


    # Plot initial seeds
    initial = results[1]
    img_vis = np.stack([initial['image']]*3, axis=-1) if len(initial['image'].shape) == 2 else initial['image']
    img_vis = img_vis.copy()
    img_vis[initial['fg_seeds']] = [1, 1, 0]  # Yellow
    img_vis[initial['bg_seeds']] = [0, 0, 1]  # Blue
    axs[0,0].imshow(img_vis)
    axs[0,0].set_title("Initial Seeds")

    # Plot initial seeds
    end_banded = results[-1]
    seg_vis = np.zeros((*end_banded['segmentation'].shape, 4))
    seg_vis[~end_banded['segmentation']] = [1, 1, 0, 0.6]  # Yellow FG
    seg_vis[end_banded['segmentation']] = [0, 0, 1, 0.6]   # Blue BG
    axs[0,1].imshow(end_banded['image'], cmap='gray' if len(end_banded['image'].shape) == 2 else None)
    axs[0,1].imshow(seg_vis)
    axs[0,1].set_title(f"end_banded\n(Time spent: {sum([s['time'] for s in seg_steps]):.2g}s | Memory: {sum([s['memory'] for s in seg_steps]):.2g}MB")

    
    # Plot coarse levels and their segmentations
    for i, seg in enumerate(seg_steps, start=2):

        if 'segmentation' in seg:
            seg_vis = np.zeros((*seg['segmentation'].shape, 4))
            seg_vis[~seg['segmentation']] = [1, 1, 0, 0.6]  # Yellow FG
            seg_vis[seg['segmentation']] = [0, 0, 1, 0.6]   # Blue BG
            
            # Overlay segmentation on original image
            axs[1,i-2].imshow(seg['image'], cmap='gray' if len(seg['image'].shape) == 2 else None)
            axs[1,i-2].imshow(seg_vis)
            
            # Add timing and memory info
            info = f"Time: {seg['time']:.2f}s\nMem: {seg['memory']:.1f}MB"
            if 'band_mask' in seg:
                info += "\n(Banded)"
                # axs[1,i].imshow(seg['band_mask'], cmap='gray' if len(seg['band_mask'].shape) == 2 else None)
            axs[1,i-2].set_title(info)
    
    # Hide unused axes
    # for j in range(i, n_cols):
    #     axs[0,j].axis('off')
    #     axs[1,j].axis('off')
    
    
    plt.suptitle("Banded Graph Cuts Progression", y=1.02)
    fig.savefig("venus_hd_res.png")
    plt.show()


# ----------------------------
# 3. Helper Functions
# ----------------------------

def regular_graph_cuts(image, fg_seeds, bg_seeds, sigma=0.1):
    """Baseline full graph cuts for comparison"""
    h, w = image.shape[:2]
    graph = maxflow.GraphFloat()
    nodeids = graph.add_grid_nodes((h, w))
    
    # Set t-links
    for y in range(h):
        for x in range(w):
            if fg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 1, 0)
            elif bg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 0, 1)
    
    # Add n-links
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    weight = np.exp(-np.sum((image[y,x]-image[ny,nx])**2/(2*sigma**2)))
                    graph.add_edge(nodeids[y, x], nodeids[ny, nx], weight, weight)
    
    graph.maxflow()
    return graph.get_grid_segments(nodeids)

# ----------------------------
# 4. Memory Measurement
# ----------------------------

def measure_memory(func, *args):
    """Proper isolated memory measurement"""
    gc.collect()
    tracemalloc.start()
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1e6  # Return MB

def display_memory_diff(stats, limit=5):
    print("\nMemory Difference:")
    for stat in stats[:limit]:
        print(f"{stat.traceback.format()[0]}: {stat.size_diff/1024:.1f} KiB")

if __name__ == "__main__":
    # Load image and seeds
    image_path = "venus_hd.jpg"
    # image_path = "venus.jpg"
    
    image = (np.array(Image.open(image_path).convert('L')) / 255.0).astype(np.float32)
    
    # Create or load labeler
    if True:  # Set to True to load previous seeds
        labeler = SeedLabeler.load_seeds(f"{image_path.split('.')[0]}_seeds")
    else:
        labeler = SeedLabeler(image, image_path)
        plt.show()  # User draws seeds
        labeler.save_seeds(f"{image_path.split('.')[0]}_seeds")  # Save after labeling
    # Compare memory usage
    # regular_seg, regular_mem = measure_memory(
    #     regular_graph_cuts, image, labeler.foreground, labeler.background)
    # print(f"Regular GC Memory: {regular_mem:.2f} MB")
    
    # seg_vis = np.zeros((*regular_seg.shape, 4))
    # seg_vis[~regular_seg] = [1, 1, 0, 0.6]  # Yellow FG
    # seg_vis[regular_seg] = [0, 0, 1, 0.6]   # Blue BG
    
    # plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    # plt.imshow(seg_vis)
    # plt.show()

    banded_seg =  memory_optimized_banded_cuts(image, labeler.foreground, labeler.background,4, 2, sigma = 0.1, factor = 2)
    seg_vis = np.zeros((*banded_seg.shape, 4))
    seg_vis[~banded_seg] = [1, 1, 0, 0.6]  # Yellow FG
    seg_vis[banded_seg] = [0, 0, 1, 0.6]   # Blue BG
    
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.imshow(seg_vis)
    plt.show()
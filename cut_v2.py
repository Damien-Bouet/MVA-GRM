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

import csv
import pandas as pd


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
                    self.foreground[int(y_):int(y_)+4, int(x_):int(x_)+4] = True
                elif self.current_label == 'background':
                    self.background[int(y_):int(y_)+4, int(x_):int(x_)+4] = True
                    
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
        else:
            print(os.path.join(save_dir, 'image_path.txt'))
        if image_path and os.path.exists(image_path):
            image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
        else:
            raise FileNotFoundError(f"Original image not found at {image_path}")
        
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

def create_memory_optimized_graph(img, level, band_width, prev_seg, sigma):
    """Memory-efficient graph construction"""
    graph = maxflow.GraphFloat()
    
    padded_seg = np.pad(prev_seg, band_width, mode='reflect')

    # Perform operations on padded array
    inner_edge = binary_erosion(padded_seg, iterations=band_width)
    outer_edge = binary_dilation(padded_seg, iterations=band_width)
    del padded_seg

    sink_region = binary_dilation(outer_edge) & ~outer_edge
    sink_pixels = np.argwhere(sink_region[band_width:-band_width, band_width:-band_width])
    source_region = ~binary_erosion(inner_edge) & inner_edge
    
    # Calculate band and remove padding
    band_mask = (outer_edge ^ inner_edge)[band_width:-band_width, band_width:-band_width]
    del outer_edge, inner_edge
    
    source_pixels = np.argwhere(source_region[band_width:-band_width, band_width:-band_width])
    band_pixels = np.argwhere(band_mask | source_region[band_width:-band_width, band_width:-band_width] | sink_region[band_width:-band_width, band_width:-band_width])
    del source_region, sink_region, band_mask

    graph.add_nodes(len(band_pixels))
    
    node_map = {(y,x): idx for idx, (y,x) in enumerate(band_pixels)}

    # 2. Process in chunks to reduce peak memory
    chunk_size = 10000
    offsets = [(-1, 0),
           (0, -1),
           (0, 1),
            (1, 0)]

    for i in range(0, len(band_pixels), chunk_size):
        chunk = band_pixels[i:i+chunk_size]
        
        # Add t-links
        for y, x in chunk:
            if np.any(np.all(sink_pixels == (y,x), axis=1)):
                graph.add_tedge(node_map[(y,x)], 1e9, 0)
            elif np.any(np.all(source_pixels == (y,x), axis=1)):
                graph.add_tedge(node_map[(y,x)], 0, 1e9)
            
            else:
                graph.add_tedge(node_map[(y,x)], 0, 0)
        
        # Add n-links
        for y, x in chunk:
            for dy, dx in offsets:
                ny, nx = y+dy, x+dx
                if (ny,nx) in node_map:
                    diff = np.sum((img[y,x]-img[ny,nx])**2)
                    # Compute weight with distance normalization
                    weight = np.exp(-diff / (2 * sigma**2))
                    graph.add_edge(node_map[(y,x)], node_map[(ny,nx)], weight, weight)
    
    return graph, node_map

# ----------------------------
# 2. Multilevel Banded Cuts
# ----------------------------

def memory_optimized_banded_cuts(image, fg_seeds, bg_seeds, levels=3, band_width=2, factor = 2, sigma=0.1, compute_baseline = False):
    """Memory-optimized version with proper visualization"""
    # Initialize results storage
    all_results = []
    #0. Baseline, regular GC
    if compute_baseline:
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
    else:
        regular_seg = None
    # 1. Coarsening stage with memory tracking
    current_img, current_fg, current_bg = image, fg_seeds, bg_seeds

    all_results.append({
        'type': 'pyramid',
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

        all_results.append({
        'type': 'pyramid',
        'image': new_img.copy(),
        'fg_seeds': new_fg.copy(),
        "bg_seeds": new_bg.copy()
        
        })
        
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
    for level in reversed(range(levels)):
        current_seg = seg.copy()        
        t0 = time()
        
        # Project segmentation
        curr_img = pyramids[level][0]
        plt.imshow(current_seg, cmap="gray")
        plt.savefig(f"seg{level}_before")
        plt.close()
        seg = cv2.resize(current_seg.copy().astype(float), curr_img.shape[:2][::-1], 
                        interpolation=cv2.INTER_NEAREST)>0.5
        plt.imshow(seg.astype(int), cmap="gray")
        plt.savefig(f"seg{level}_after")    
        plt.close()
        # Create band
        band_mask = (binary_dilation(seg, iterations=band_width//2) ^ 
                     binary_erosion(seg, iterations=band_width//2 + 1))
        
        plt.imshow(band_mask, cmap="gray")
        plt.savefig(f"band_mask{level}_after")
        plt.close()
        # Build and solve banded graph
        gc.collect()
        tracemalloc.start()
        print("Creating the Graph...")
        graph, node_map = create_memory_optimized_graph(curr_img, level, band_width, seg, sigma)
        print("Doing max flow...")
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



    
    return regular_seg, seg, all_results

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
    baseline = [r for r in results if r['type'] == "baseline"]
    pyramids = [r for r in results if r['type'] == "pyramid"]
    # Create figure layout
    n_cols = len(seg_steps)
    fig, axs = plt.subplots(2, n_cols, 
                          figsize=(n_cols*4*aspect_ratio, 2.5*4),
                          constrained_layout=True)
    
    #Plot baseline
    for b in baseline: 
        end_seeds = pyramids[0]
        seg_vis = np.zeros((*b['segmentation'].shape, 4))
        seg_vis[b['segmentation']] = [0.8, 0.4, 0, 0.4]  # Yellow FG
        seg_vis[~b['segmentation']] = [0, 0, 0.8, 0.4]   # Blue BG
        seg_vis[end_seeds['fg_seeds']] = np.array([1, 1, 0, 1])  # Yellow
        seg_vis[end_seeds['bg_seeds']] = np.array([0, 0, 1, 1])  # Blue
        axs[0,2].imshow(b['image'], cmap='gray' if len(b['image'].shape) == 2 else None)
        axs[0,2].imshow(seg_vis)
        axs[0,2].set_title(f"Baseline\nTime spent: {b['time']:.3g}s\nMemory: {b['memory']:.3g}MB")
    # else:
    #     axs[0,2].axis('off')

    # Plot initial seeds
    initial = pyramids[0]
    img_vis = np.stack([initial['image']]*3, axis=-1) if len(initial['image'].shape) == 2 else initial['image']
    img_vis = img_vis.copy()
    img_vis = np.concatenate([img_vis, np.ones(img_vis.shape[:2])[:, :, None]], axis=2)
    img_vis[initial['fg_seeds']] = [1, 1, 0, 1]  # Yellow
    img_vis[initial['bg_seeds']] = [0, 0, 1, 1]  # Blue
    axs[0,0].imshow(img_vis)
    axs[0,0].set_title("Initial Seeds")

    # Plot initial seeds
    end_banded = seg_steps[-1]
    end_seeds = pyramids[0]
    seg_vis = np.zeros((*end_banded['segmentation'].shape, 4))
    seg_vis[end_banded['segmentation']] = [0.8, 0.4, 0, 0.4]  # Yellow FG
    seg_vis[~end_banded['segmentation']] = [0, 0, 0.8, 0.4]   # Blue BG
    seg_vis[end_seeds['fg_seeds']] = [1, 1, 0, 1]  # Yellow
    seg_vis[end_seeds['bg_seeds']] = [0, 0, 1, 1]  # Blue
    axs[0,1].imshow(end_banded['image'], cmap='gray' if len(end_banded['image'].shape) == 2 else None)
    axs[0,1].imshow(seg_vis)
    axs[0,1].set_title(f"End Banded GC\nTime spent: {sum([s['time'] for s in seg_steps]):.2g}s\nMemory: {max([s['memory'] for s in seg_steps]):.2g}MB")

    for k in range(2, n_cols):
        axs[0,k].axis("off")
    
    # Plot coarse levels and their segmentations
    for i, seg in enumerate(seg_steps, start=0):
        if 'segmentation' in seg:            
            # Overlay segmentation on original image
            # print(len(pyramids))
            corresponding_seed = pyramids[-1 - i]
            # print(corresponding_seed["image"].shape)

            seg_vis = np.zeros((*seg['segmentation'].shape, 4))
            seg_vis[seg['segmentation']] = [0.8, 0.4, 0, 0.4]  # Yellow FG
            seg_vis[~seg['segmentation']] = [0, 0, 0.8, 0.4]   # Blue BG

            img_vis = np.stack([corresponding_seed['image']]*3, axis=-1) if len(corresponding_seed['image'].shape) == 2 else corresponding_seed['image']
            img_vis = img_vis.copy()
            img_vis = np.concatenate([img_vis, np.ones(img_vis.shape[:2])[:, :, None]], axis=2)
            # img_vis[seg['segmentation']] += np.array([0.8, 0.4, 0, 0.4])  # Yellow FG
            # img_vis[~seg['segmentation']] += np.array([0, 0, 0.8, 0.4])   # Blue BG
            
            # Add timing and memory info
            info = f"Time: {seg['time']:.2f}s\nMem: {seg['memory']:.1f}MB"
            if 'band_mask' in seg:
                info += "\n(Banded)"
                seg_vis[seg["band_mask"]] = [0, 0, 0, 0.2]
                # axs[1,i].imshow(seg['band_mask'], cmap='gray' if len(seg['band_mask'].shape) == 2 else None)

            seg_vis[corresponding_seed['fg_seeds']] = [1, 1, 0, 1]  # Yellow
            seg_vis[corresponding_seed['bg_seeds']] = [0, 0, 1, 1]  # Blue
            
            axs[1,i].imshow(img_vis)
            axs[1,i].imshow(seg_vis)
            axs[1, i].set_title(f"Coarse {n_cols-1-i} - {seg['time']:.2g}s - {seg['memory']:.2g}MB")
            # axs[1,i].imshow(seg_vis)
    # Hide unused axes
    for j in range(0, n_cols):
        axs[0,j].axis('off')
        axs[1,j].axis('off')
    
    
    plt.suptitle("Banded Graph Cuts Progression")
    fig.savefig(os.path.join(os.path.dirname(IMAGE_PATH), "results", os.path.basename(IMAGE_PATH),"all_res.png"))
    plt.show()


# ----------------------------
# 3. Helper Functions
# ----------------------------

def regular_graph_cuts(image, fg_seeds, bg_seeds, sigma=0.1):
    """Baseline full graph cuts for comparison"""
    h, w = image.shape[:2]
    graph = maxflow.GraphFloat()
    nodeids = graph.add_grid_nodes((h, w))
    
    print("Creating the Graph...")
    # Set t-links
    for y in range(h):
        for x in range(w):
            if fg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 0, 1e9)
            elif bg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 1e9, 0)
    
    # Add n-links
    offsets = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),        (0, 1),
           (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    weight = np.exp(-np.sum((image[y,x]-image[ny,nx])**2/(2*sigma**2)))
                    graph.add_edge(nodeids[y, x], nodeids[ny, nx], weight, weight)
    
    print("Doing max flow...")
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

def get_metrics(segmentation, ground_truth):
    TP = np.sum(np.logical_and(segmentation == 1, ground_truth == 1))
    FP = np.sum(np.logical_and(segmentation == 1, ground_truth == 0))
    TN = np.sum(np.logical_and(segmentation == 0, ground_truth == 0))
    FN = np.sum(np.logical_and(segmentation == 0, ground_truth == 1))

    recall = TP/(TP + FN) if TP + FN >0 else 1

    precision = TP/(TP+FP) if TP + FP >0 else 1

    F1_score = 2*recall*precision / (recall + precision)

    DICE = 2*TP/(2*TP + FP + FN)

    accuracy = (TP+TN)/(segmentation.shape[0] * segmentation.shape[1])
    return DICE, recall, precision



if __name__ == "__main__":
    # Load image and seeds
    IMAGE_PATH = "images/car_4k.jpg"
    GROUND_TRUTH = "images/car_4k_gt.png"

    results_path = os.path.join(os.path.dirname(IMAGE_PATH), "results", os.path.basename(IMAGE_PATH))
    os.makedirs(results_path, exist_ok=True)

    # Load images
    image = (np.array(Image.open(IMAGE_PATH).convert('L')) / 255.0).astype(np.float32)
    true_mask = (np.array(Image.open(GROUND_TRUTH).convert('L')) / 255.0).astype(np.float32)
    true_mask[true_mask > 0] = 1
    true_mask = true_mask.astype(bool)

    # Create or load labeler
    if False:  # Set to True to load previous seeds
        labeler = SeedLabeler.load_seeds(f"{IMAGE_PATH.split('.')[0]}_seeds")
    else:
        labeler = SeedLabeler(image, IMAGE_PATH)
        plt.show()  # User draws seeds
        labeler.save_seeds(f"{IMAGE_PATH.split('.')[0]}_seeds")  # Save after labeling

    compute_baseline = False

    # CSV file setup
    csv_path = os.path.join(results_path, "experiment_results.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Bandwidth", "Level", "Factor", "F1_Baseline", "Time_Baseline", "Memory_Baseline",
                         "F1_Banded", "Time_Banded", "Memory_Banded"])

    # Main experiment loop
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        for level in [5]:
            for factor in [2]:
                for bandwidth in [2]:
                    for sigma in [0.1]:
                        if level * factor <= 16:
                            baseline_seg, banded_seg, all_results = memory_optimized_banded_cuts(
                                image, labeler.foreground, labeler.background, level,
                                band_width=bandwidth, sigma=sigma, factor=factor, compute_baseline=compute_baseline)
                            
                            # 4. Create memory-efficient visualization
                            create_detailed_visualization(all_results, image.shape[1]/image.shape[0])
            
                            # # Initialize values
                            # DICE_baseline, time_baseline, memory_baseline = None, None, None

                            # # Process baseline segmentation (if computed)
                            # if baseline_seg is not None:
                            #     DICE_baseline, recall, precision = get_metrics(baseline_seg, true_mask)
                            #     time_baseline = 1 # all_results[0]["time"]
                            #     memory_baseline = 2 # all_results[0]["memory"]

                            #     # Save baseline segmentation image
                            #     seg_vis = np.zeros((*baseline_seg.shape, 4))
                            #     seg_vis[~baseline_seg] = [1, 1, 0, 0.6]  # Yellow FG
                            #     seg_vis[baseline_seg] = [0, 0, 1, 0.6]   # Blue BG
                            #     plt.imshow(image, cmap="grey")
                            #     plt.imshow(seg_vis, alpha=0.5)
                            #     plt.tight_layout()
                            #     plt.axis("off")
                            #     plt.savefig(os.path.join(results_path, f"regular_gc_results_{DICE_baseline:.4f}_{time_baseline:.2g}s_{memory_baseline:.3g}MB.png"))
                            #     compute_baseline = False  # Compute baseline only once

                            # # Process banded segmentation
                            # accuracy, recall, precision = get_metrics(banded_seg, true_mask)
                            # time_banded = sum([s['time'] for s in all_results if s['type'] in ['segmentation', 'refinement']])
                            # memory_banded = max([s['memory'] for s in all_results if s['type'] in ['segmentation', 'refinement']])

                            # # Save banded segmentation image
                            # seg_vis = np.zeros((*banded_seg.shape, 4))
                            # seg_vis[~banded_seg] = [1, 1, 0, 0.6]  # Yellow FG
                            # seg_vis[banded_seg] = [0, 0, 1, 0.6]   # Blue BG
                            # plt.imshow(image, cmap="grey")
                            # plt.imshow(seg_vis, alpha=0.5)
                            # plt.tight_layout()
                            # plt.axis("off")
                            # # plt.savefig(os.path.join(results_path, f"{bandwidth}_{level}_{factor}_{sigma}_banded_results_{accuracy:.4f}_{time_banded:.2g}s_{memory_banded:.3g}MB.png"))
                            # plt.show()
                            # # Save results to CSV
                            # writer.writerow([bandwidth, level, factor, DICE_baseline, time_baseline, memory_baseline,
                            #                 accuracy, time_banded, memory_banded])
                        

    # df = pd.read_csv(csv_path)
    # plt.plot(df["Bandwidth"], df["Memory_Banded"], marker="o", label="Banded Memory Usage")
    # plt.xlabel("Bandwidth")
    # plt.ylabel("Memory (MB)")
    # plt.legend()
    # plt.show()



                
    
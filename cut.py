import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
import maxflow
import os
from time import time
from PIL import Image
from tqdm import tqdm
# ----------------------------
# 1. Enhanced SeedLabeler with Save/Load
# ----------------------------

class SeedLabeler:
    def __init__(self, image, image_path=None, no_fig=False):
        self.image = image
        self.image_path = image_path  # Store image path for reloading
        self.foreground = np.zeros((image.shape[0], image.shape[1]), dtype=float)
        self.background = np.zeros((image.shape[0], image.shape[1]), dtype=float)
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
                    self.foreground[int(y_):int(y_)+2, int(x_):int(x_)+2] = 1
                elif self.current_label == 'background':
                    self.background[int(y_):int(y_)+2, int(x_):int(x_)+2] = 1
                    
        if self.current_label == 'foreground':
            self.foreground[y:y+2, x:x+2] = 1
        elif self.current_label == 'background':
            self.background[y:y+2, x:x+2] = 1
            
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
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
        else:
            raise FileNotFoundError("Original image not found")
        
        labeler = cls(image, image_path, no_fig=True)
        labeler.foreground = np.load(os.path.join(save_dir, 'foreground.npy'))
        labeler.background = np.load(os.path.join(save_dir, 'background.npy'))
        return labeler


# ----------------------------
# 2. Fixed Coarsening Functions
# ----------------------------
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
    coarse = padded.reshape(h_p//factor, factor, w_p//factor, factor).max(axis=(1, 3))
    return coarse[:h//factor, :w//factor]  # Crop back to original size

# ----------------------------
# 3. Rest of Pipeline (Same as Before)
# ----------------------------
def create_graph(image, fg_seeds, bg_seeds, sigma=0.1, connectivity=4):
    h, w = image.shape[:2]
    graph = maxflow.GraphFloat()
    nodeids = graph.add_grid_nodes((h, w))
    
    # Add t-links
    for y in range(h):
        for x in range(w):
            if fg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 1e9, 0)
            elif bg_seeds[y, x]:
                graph.add_tedge(nodeids[y, x], 0, 1e9)
            else:
                graph.add_tedge(nodeids[y, x], 0, 0)
    
    # Add n-links with proper connectivity
    if connectivity == 4:
        structure = maxflow.vonNeumann_structure()
    else:  # 8-connectivity
        structure = maxflow.moore_structure()
    
    # Create offset list from structure
    offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            if structure[dy+1, dx+1]:
                offsets.append((dy, dx))
    
    # Add edges for all valid offsets
    for y in tqdm(range(h)):
        for x in range(w):
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    diff = np.linalg.norm(image[y, x] - image[ny, nx])
                    weight = np.exp(-diff**2/(2*sigma**2))
                    graph.add_edge(nodeids[y, x], nodeids[ny, nx], weight, weight)
    
    return graph, nodeids

# ----------------------------
# 4. Regular Graph Cuts Baseline
# ----------------------------
def regular_graph_cuts(image, fg_seeds, bg_seeds, sigma=0.1, connectivity=8):
    """Standard graph cuts without multilevel optimization"""
    graph, nodeids = create_graph(image, fg_seeds, bg_seeds, sigma, connectivity)
    graph.maxflow()
    return graph.get_grid_segments(nodeids)


# # ----------------------------
# # 4. Multilevel Banded Cuts
# # ----------------------------
# def multilevel_banded_cuts(image, fg_seeds, bg_seeds, levels=2, band_width=1, sigma=0.1, connectivity=8):
#     # Coarsening stage
#     pyramids = [(image, fg_seeds, bg_seeds)]
#     for _ in range(levels-1):
#         c_img = coarsen(pyramids[-1][0])
#         c_fg = coarsen_seeds(pyramids[-1][1])
#         c_bg = coarsen_seeds(pyramids[-1][2])
#         pyramids.append((c_img, c_fg, c_bg))
    
#     # Initial segmentation at coarsest level
#     c_img, c_fg, c_bg = pyramids[-1]
    
#     graph, nodeids = create_graph(c_img, c_fg, c_bg, sigma, connectivity)
#     graph.maxflow()
#     segmentation = graph.get_grid_segments(nodeids)

#     r = c_img.shape[0]/c_img.shape[1]
#     fig, ax = plt.subplots(1, 2, figsize=(2*5*r, 5*r))
#     imshow = np.copy(c_img)
#     imshow[c_fg.astype(bool)] = [1, 1, 0]
#     imshow[c_bg.astype(bool)] = [0, 0, 1]
#     ax[0].imshow(imshow)
#     ax[0].set_title("Original Image")
#     ax[0].axis("off")

#     seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
#     seg[segmentation==1] = 1
#     seg[c_fg.astype(bool)] = [1, 1, 0]  # Yellow
#     seg[c_bg.astype(bool)] = [0, 0, 1]  # Blue
#     ax[1].imshow(seg)
#     ax[1].set_title("Regular Graph Cuts")
#     ax[1].axis("off")
#     plt.show()
    
#     # Uncoarsening
#     for level in reversed(range(levels-1)):
#         # Project segmentation to next level
#         curr_seg = segmentation.astype(np.uint8)
#         curr_seg = cv2.resize(curr_seg, pyramids[level][0].shape[:2][::-1], 
#                             interpolation=cv2.INTER_NEAREST)
        
#         # Create narrow band
#         band = binary_dilation(curr_seg, iterations=band_width)
#         band = binary_dilation(band, iterations=band_width)  # Expand both sides
        
#         # Get current level data
#         img, fg, bg = pyramids[level]
#         new_fg = np.logical_and(fg, band)
#         new_bg = np.logical_and(bg, band)
            
#         # Create banded graph
#         graph, nodeids = create_graph(img, new_fg, new_bg)
#         graph.maxflow()
#         segmentation = graph.get_grid_segments(nodeids)
    
#         r = img.shape[0]/img.shape[1]
#         fig, ax = plt.subplots(1, 2, figsize=(2*5*r, 5*r))
#         imshow = np.copy(img)
#         imshow[fg.astype(bool)] = [1, 1, 0]
#         imshow[bg.astype(bool)] = [0, 0, 1]
#         ax[0].imshow(imshow)
#         ax[0].set_title("Original Image")
#         ax[0].axis("off")

#         seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
#         seg[segmentation==1] = 1
#         seg[new_fg.astype(bool)] = [1, 1, 0]  # Yellow
#         seg[new_bg.astype(bool)] = [0, 0, 1]  # Blue
#         ax[1].imshow(seg)
#         ax[1].set_title("Regular Graph Cuts")
#         ax[1].axis("off")
#         plt.show()

#     return segmentation


def create_band_and_seeds(curr_seg, band_width=1):
    """Create narrow band and define new seeds as per paper specifications"""
    # Create band
    dilated = binary_dilation(curr_seg, iterations=band_width)
    eroded = binary_erosion(curr_seg, iterations=band_width)
    band = dilated ^ eroded  # XOR to get the band
    
    # Define new seeds: everything OUTSIDE the band keeps its previous segmentation
    new_fg_seeds = np.logical_and(~curr_seg, ~band)  # Inside curr_seg but outside band
    new_bg_seeds = np.logical_and(curr_seg, ~band)  # Outside curr_seg and outside band
    
    return band, new_fg_seeds, new_bg_seeds

def multilevel_banded_cuts(image, fg_seeds, bg_seeds, levels=2, band_width=1, sigma=0.1, connectivity=8, factor=2):
    # Coarsening stage
    all_results = []
    t0 = time()
    
    # Coarsening stage
    pyramids = [(image, fg_seeds, bg_seeds)]
    for _ in tqdm(range(levels-1)):
        c_img = coarsen(pyramids[-1][0], factor)
        c_fg = coarsen_seeds(pyramids[-1][1], factor)
        c_bg = coarsen_seeds(pyramids[-1][2], factor)
        pyramids.append((c_img, c_fg, c_bg))
    
    # Initial segmentation at coarsest level
    c_img, c_fg, c_bg = pyramids[-1]
    graph, nodeids = create_graph(c_img, c_fg, c_bg, sigma, connectivity)
    t0 = time()
    graph.maxflow()
    t1 = time()
    segmentation = graph.get_grid_segments(nodeids)
    
    # Store coarsest level results
    all_results.append({
        "time": t1 - t0,
        'level': levels-1,
        'image': c_img,
        'segmentation': segmentation,
        'prev_seg': None,
        'fg_seeds': c_fg,
        'bg_seeds': c_bg
    })
    
    # Uncoarsening
    for level in tqdm(reversed(range(levels-1))):
        # Project segmentation to next level
        curr_seg = segmentation
        curr_seg = cv2.resize(curr_seg.astype(np.uint8), pyramids[level][0].shape[:2][::-1], 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Create band and new seeds
        band, new_fg_seeds, new_bg_seeds = create_band_and_seeds(curr_seg, band_width)
        
        # Create graph for current level
        img = pyramids[level][0]
        print(img.shape)
        graph, nodeids = create_graph(img, new_fg_seeds, new_bg_seeds, sigma, connectivity)
        t0 = time()
        graph.maxflow()
        t1 = time()
        segmentation = graph.get_grid_segments(nodeids)
        
        # Store results for this level
        all_results.append({
            "time": t1 - t0,
            'level': level,
            'image': img,
            'segmentation': segmentation,
            'prev_seg': curr_seg,
            'fg_seeds': new_fg_seeds,
            'bg_seeds': new_bg_seeds
        })

    # banded_time = time() - t0
    
    
    # Compute regular graph cut for comparison
    graph, nodeids = create_graph(pyramids[0][0], pyramids[0][1], pyramids[0][2], sigma, connectivity)
    t0 = time()
    graph.maxflow()
    regular_time = time() - t0
    regular_seg = graph.get_grid_segments(nodeids)

    # Create final visualization
    r = image.shape[1] / image.shape[0]
    n_rows = levels  # Initial + all levels + comparison
    fig, axs = plt.subplots(2, max(3, n_rows), figsize=(4*max(3, n_rows)*r, 8))
    
    for ax in axs.flatten():
        ax.axis("off")

    # Plot initial image with seeds
    orig_img = pyramids[0][0]
    orig_fg = pyramids[0][1]
    orig_bg = pyramids[0][2]
    img_vis = np.copy(orig_img)
    if len(img_vis.shape) == 2:
            img_vis = np.concatenate([img_vis[:, :, None]]*3, axis=2)
    img_vis[orig_fg.astype(bool)] = [1, 1, 0]  # Yellow for initial FG seeds
    img_vis[orig_bg.astype(bool)] = [0, 0, 1]  # Blue for initial BG seeds
    axs[0, 0].imshow(img_vis)
    axs[0, 0].set_title("Initial Image\nwith Seeds")

    # Plot regular graph cut result
    seg_vis = np.zeros((*segmentation.shape, 3))
    seg_vis[~segmentation] = [1, 1, 0]  # Yellow for FG
    seg_vis[segmentation] = [0, 1, 1]  # Cyan for BG
    axs[0, 1].imshow(img_vis)
    axs[0, 1].imshow(seg_vis, alpha=0.6)
    axs[0, 1].set_title(f"Banded Graph Cut ({sum([res['time'] for res in all_results]):.2g}s)")

    # Plot regular graph cut result
    # reg_vis = np.zeros((*regular_seg.shape, 3))
    # reg_vis[~regular_seg] = [1, 1, 0]  # Yellow for FG
    # reg_vis[regular_seg] = [0, 1, 1]  # Cyan for BG
    # axs[0, 2].imshow(img_vis)
    # axs[0, 2].imshow(reg_vis, alpha=0.6)
    # axs[0, 2].set_title(f"Regular\nGraph Cut ({regular_time:.2g}s)")
    
    # Plot intermediate results
    for i, result in enumerate(all_results):
        if i == 0:
            continue
        seg_vis = np.zeros((*result['segmentation'].shape, 4))
        
        seg_vis[result['fg_seeds']] = [1, 1, 0, 1]  # [0.8, 0.4, 0, 1]  # Dark orange for previous FG
        seg_vis[result['bg_seeds']] = [0, 1, 1, 1]  #[0, 0, 0.8, 1]  # Dark blue for previous BG
        
        axs[1, i-1].imshow(result["image"])
        axs[1, i-1].imshow(seg_vis, alpha=0.6)
        axs[1, i-1].set_title(f"Level {result['level']+1}\nSegmentation: {result['time']:.2g}s")
    
    result = all_results[-1]
    seg_vis = np.zeros((*result['segmentation'].shape, 4))    
    seg_vis[~result['segmentation']] = [1, 1, 0, 1]  # [0.8, 0.4, 0, 1]  # Dark orange for previous FG
    seg_vis[result['segmentation']] = [0, 1, 1, 1]  #[0, 0, 0.8, 1]  # Dark blue for previous BG
    
    axs[1, -1].imshow(result["image"])
    axs[1, -1].imshow(seg_vis, alpha=0.6)
    axs[1, -1].set_title(f"Level {result['level']}\nSegmentation: {result['time']:.2g}s")
    
    plt.tight_layout()
    plt.show()
    
    return segmentation
    
if __name__ == "__main__":
    # Load image
    image_path = "venus.jpg"
    
    image = np.array(Image.open(image_path).convert('L')) / 255.0
    
    # Create or load labeler
    if True:  # Set to True to load previous seeds
        labeler = SeedLabeler.load_seeds(f"{image_path.split('.')[0]}_seeds")
    else:
        labeler = SeedLabeler(image, image_path)
        plt.show()  # User draws seeds
        labeler.save_seeds(f"{image_path.split('.')[0]}_seeds")  # Save after labeling
    
    # # Run both algorithms
    # print("Running regular graph cuts...")
    # regular_seg = regular_graph_cuts(image, labeler.foreground, labeler.background, 0.1, 8)
    
    print("Running multilevel banded graph cuts...")
    banded_seg = multilevel_banded_cuts(
        image, 
        labeler.foreground, 
        labeler.background,
        levels=4,
        band_width=1,
        factor=2,
        # sigma=0.3,
    )
    
    # r = image.shape[0]/image.shape[1]
    # # Show comparison
    # fig, ax = plt.subplots(1, 3, figsize=(3*5*r, 5*r))
    # ax[0].imshow(image)
    # ax[0].set_title("Original Image")
    # ax[0].axis("off")

    # seg = np.zeros((regular_seg.shape[0], regular_seg.shape[1], 3))
    # seg[regular_seg==1] = 1
    # seg[labeler.foreground.astype(bool)] = [1, 1, 0]  # Yellow
    # seg[labeler.background.astype(bool)] = [0, 0, 1]  # Blue
    # ax[1].imshow(seg)
    # ax[1].set_title("Regular Graph Cuts")
    # ax[1].axis("off")
    
    # ax[2].imshow(banded_seg, cmap='gray')
    # ax[2].set_title("Multilevel Banded Cuts")
    # ax[2].axis("off")
    
    # plt.tight_layout()
    # plt.show()
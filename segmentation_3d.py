import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion
import maxflow
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class SeedLabeler:
    def __init__(self, image, slice_index=0, image_path=None, no_fig=False):
        self.image = image  # Expecting a 3D array
        self.slice_index = slice_index
        self.image_path = image_path  # Store image path for reloading
        self.foreground = np.zeros(image.shape, dtype=bool)  # Full 3D mask
        self.background = np.zeros(image.shape, dtype=bool)  # Full 3D mask
        self.drawing = False  # Track whether mouse button is held
        self.current_label = None  # Track foreground or background labeling
        self.previous_pos = None
        
        if not no_fig:
            self.fig, self.ax = plt.subplots()
            self.update_display()
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
        slice_fg = self.foreground[:, :, self.slice_index]
        slice_bg = self.background[:, :, self.slice_index]
        
        if self.previous_pos is not None:
            n = max(abs(self.previous_pos[0] - x), abs(self.previous_pos[1] - y))
            for x_, y_ in zip(np.linspace(self.previous_pos[0], x, n), np.linspace(self.previous_pos[1], y, n)):
                if self.current_label == 'foreground':
                    slice_fg[int(y_):int(y_)+4, int(x_):int(x_)+4] = True
                elif self.current_label == 'background':
                    slice_bg[int(y_):int(y_)+4, int(x_):int(x_)+4] = True
                    
        if self.current_label == 'foreground':
            slice_fg[y:y+2, x:x+2] = True
        elif self.current_label == 'background':
            slice_bg[y:y+2, x:x+2] = True
            
        self.previous_pos = (x, y)
        self.update_display()
    
    def update_display(self):
        print(self.slice_index)
        slice_image = self.image[:, :, self.slice_index]
        disp = np.stack([slice_image]*3, axis=-1) if len(slice_image.shape) == 2 else slice_image.copy()
        
        disp[self.foreground[:, :, self.slice_index].astype(bool)] = [1, 1, 0]  # Yellow
        disp[self.background[:, :, self.slice_index].astype(bool)] = [0, 0, 1]  # Blue
        self.ax.imshow(disp, cmap="gray")
        self.fig.canvas.draw()
    
    def save_seeds(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'foreground.npy'), self.foreground)
        np.save(os.path.join(save_dir, 'background.npy'), self.background)
        if self.image_path:
            with open(os.path.join(save_dir, 'image_path.txt'), 'w') as f:
                f.write(self.image_path)
    
    @classmethod
    def load_seeds(cls, save_dir):
        image_path = None
        if os.path.exists(os.path.join(save_dir, 'image_path.txt')):
            with open(os.path.join(save_dir, 'image_path.txt'), 'r') as f:
                image_path = f.read().strip()
        
        if image_path and os.path.exists(image_path):
            nii = nib.load(VOXEL_PATH)
            volume = nii.get_fdata() #np array

            volume -= np.min(volume)
            volume = volume / np.max(volume)
            volume = np.transpose(volume, (2, 1, 0))
        else:
            raise FileNotFoundError(f"Original image not found at {image_path}")
        
        labeler = cls(volume, no_fig=True)
        labeler.foreground = np.load(os.path.join(save_dir, 'foreground.npy')).astype(bool)
        labeler.background = np.load(os.path.join(save_dir, 'background.npy')).astype(bool)
        return labeler


FG_PATH = "slice_120_seeds/foreground.npy"
BG_PATH = "slice_120_seeds/background.npy"
VOXEL_PATH = "IMG_0002.nii.gz"  # la_030_scan.nii "image1/liver_1.nii"
GT_PATH = "IMG_0002.nii.gz"  # la_030_gt.nii


nii = nib.load(VOXEL_PATH)
volume = nii.get_fdata() #np array
nii_label = nib.load(GT_PATH)
volume_label = nii_label.get_fdata() #np array

volume -= np.min(volume)
volume = volume / np.max(volume)
volume = np.transpose(volume, (2, 1, 0))
print(np.min(volume), np.max(volume))


slice_idx = 100

if True:  # Set to True to load previous seeds
    labeler = SeedLabeler.load_seeds(f"{VOXEL_PATH.split('.')[0]}_seeds")
else:
    labeler = SeedLabeler(volume, slice_idx, VOXEL_PATH)
    plt.show()  # User draws seeds
    labeler.save_seeds(f"{VOXEL_PATH.split('.')[0]}_seeds")  # Save after labeling

fg_3d = labeler.foreground
bg_3d = labeler.background

# for n in [slice_idx]:
#     plt.imshow(volume[:, :, n], cmap="gray")
#     plt.imshow(fg_3d[:, :, n], cmap="Blues", alpha=0.2)
#     plt.imshow(bg_3d[:, :, n], cmap="Reds", alpha=0.2)
#     plt.imshow(volume_label[:, :, n], cmap="Greens", alpha=0.2)
#     plt.show()

# fg_expanded = expand_foreground_seed(fg_3d, radius=1)

# seg_sub = regular_graph_cuts_3d(subvol, fg_expanded[:, :, z_start:z_end], bg_3d[:, :, z_start:z_end], sigma=5)

# np.save("regular_segmentation_3d_mask.npy", seg_sub.astype(np.uint8))



# # #graph cuts
# def regular_graph_cuts_3d(volume, fg_seeds, bg_seeds, sigma=0.1):
#     print("DEBUG SHAPES:")
#     print("volume.shape:", volume.shape)
#     print("fg_seeds.shape:", fg_seeds.shape)
#     print("bg_seeds.shape:", bg_seeds.shape)

#     assert volume.shape == fg_seeds.shape == bg_seeds.shape, "Shape mismatch!"
    
#     D, H, W = volume.shape
#     graph = maxflow.GraphFloat()
#     nodeids = graph.add_grid_nodes((D, H, W))
#     print("Creating the Graph...")

#     # Add t-links
#     for z in range(D):
#         for y in range(H):
#             for x in range(W):
#                 if fg_seeds[z, y, x]:
#                     graph.add_tedge(nodeids[z, y, x], 0, 1e9)
#                 elif bg_seeds[z, y, x]:
#                     graph.add_tedge(nodeids[z, y, x], 1e9, 0)

#     # 6-connected neighborhood
#     offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
#     for z in range(D):
#         for y in range(H):
#             for x in range(W):
#                 for dz, dy, dx in offsets:
#                     nz, ny, nx = z + dz, y + dy, x + dx
#                     if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
#                         diff = (volume[z, y, x] - volume[nz, ny, nx]) ** 2
#                         weight = np.exp(-diff / (2 * sigma ** 2))
#                         graph.add_edge(nodeids[z, y, x], nodeids[nz, ny, nx], weight, weight)
#     print("doing maxflow")
#     graph.maxflow()
#     return graph.get_grid_segments(nodeids)

# def expand_foreground_seed(fg_3d, radius=1):
#     # Create spherical structuring element
#     zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
#     mask = xx**2 + yy**2 + zz**2 <= radius**2

#     fg_dilated = binary_dilation(fg_3d, structure=mask)
#     return fg_dilated


# #experiment
# def resize_3d(seg, new_shape):
#     """Resize a 3D segmentation mask using nearest neighbor interpolation."""
#     d, h, w = seg.shape
#     new_d, new_h, new_w = new_shape

#     # Resize height and width using cv2
#     resized_slices = [cv2.resize(seg[i, :, :].astype(float), (new_w, new_h), interpolation=cv2.INTER_NEAREST) 
#                       for i in range(d)]
#     resized_2d = np.stack(resized_slices, axis=0)  # Keep depth axis intact

#     # Resize depth using scipy.ndimage.zoom
#     scale_d = new_d / d
#     resized_3d = zoom(resized_2d, (scale_d, 1, 1), order=0)  # Nearest neighbor

#     return resized_3d > 0.5  # Thresholding
# #3d segmentation

# #3d banded cut
# def coarsen_3d(img, factor=2):
#     """Coarsen 3D image with padding if needed along all dimensions."""
#     d, h, w = img.shape[:3]
    
#     # Pad if dimensions aren't divisible by factor
#     pad_d = (factor - d % factor) % factor
#     pad_h = (factor - h % factor) % factor
#     pad_w = (factor - w % factor) % factor
    
#     if pad_d > 0 or pad_h > 0 or pad_w > 0:
#         img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    
#     # Reshape and apply downsampling along all dimensions
#     d_p, h_p, w_p = img.shape[:3]
#     img = img.reshape(d_p//factor, factor, h_p//factor, factor, w_p//factor, factor, -1)
#     img = img.mean(axis=(1, 3, 5))[:, :, :, 0]
    
#     return img[:d//factor, :h//factor, :w//factor]



# def coarsen_seeds_3d(seeds, factor=2):
#     """Coarsen 3D seeds with padding to handle non-divisible dimensions."""
#     d, h, w = seeds.shape
    
#     # Pad to make dimensions divisible by factor
#     pad_d = (factor - d % factor) % factor
#     pad_h = (factor - h % factor) % factor
#     pad_w = (factor - w % factor) % factor
    
#     padded = np.pad(seeds, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
#     print(padded.shape, factor)
#     # Reshape and take max over blocks
#     d_p, h_p, w_p = padded.shape
#     coarse = padded.reshape(d_p//factor, factor, h_p//factor, factor, w_p//factor, factor)
#     coarse = coarse.max(axis=(1, 3, 5)).astype(bool)
    
#     return coarse[:d//factor, :h//factor, :w//factor]


# def banded_cuts_3d(volume, fg_seeds, bg_seeds, levels=2, band_width=2, factor=2, sigma=0.1):
#     all_results = []
#     pyramids = [(volume.copy(), fg_seeds.copy(), bg_seeds.copy())]

#     # 1. Coarsen volume & seeds
#     for level in range(levels):
#         v, f, b = pyramids[-1]
#         v_c = coarsen_3d(v, factor=factor)
#         f_c = coarsen_seeds_3d(f.astype(np.uint8), factor=factor).astype(bool)
#         b_c = coarsen_seeds_3d(b.astype(np.uint8), factor=factor).astype(bool)
#         pyramids.append((v_c, f_c, b_c))

#     # 2. Solve coarsest
#     current_img, current_fg, current_bg = pyramids[-1]
#     #seg = regular_graph_cuts_3d(current_img, current_fg, current_bg, sigma)
#     print(current_img.shape)
#     seg = regular_graph_cuts_3d(current_img, current_fg, current_bg, sigma)
#     # (D, H, W) apparemment convention différente

#     # 3. Uncoarsening loop
#     for level in reversed(range(levels)):
#         current_seg = seg.copy()   
#         new_img, _, _ = pyramids[level]
#         # Resize segmentation to next finer level
#         # depth = new_img.shape[2]
#         # resized = []
#         # for i in range(depth):
#         #     resized_slice = cv2.resize(prev_seg[:, :, i], 
#         #                                (new_img.shape[1], new_img.shape[0]), 
#         #                                interpolation=cv2.INTER_NEAREST)
#         #     resized.append(resized_slice)
#         # seg = np.stack(resized, axis=2).astype(bool)
#         print(current_seg.shape, new_img.shape)
#         seg = resize_3d(current_seg.copy().astype(float), new_img.shape)

#         band_mask = binary_dilation(seg, iterations=band_width) ^ binary_erosion(seg, iterations=band_width)
#         band_voxels = np.argwhere(band_mask)

#         graph = maxflow.GraphFloat()
#         node_map = {}
#         nodeids = graph.add_nodes(len(band_voxels))
#         for idx, (x, y, z) in enumerate(band_voxels):
#             node_map[(x, y, z)] = idx
#             # T-links
#             if fg_seeds[x, y, z]:
#                 graph.add_tedge(idx, 0, 1e9)
#             elif bg_seeds[x, y, z]:
#                 graph.add_tedge(idx, 1e9, 0)
#             else:
#                 graph.add_tedge(idx, 0, 0)

#         # N-links
#         offsets = [(1,0,0), (0,1,0), (0,0,1)]
#         for (x, y, z), idx in node_map.items():
#             for dx, dy, dz in offsets:
#                 nx, ny, nz = x+dx, y+dy, z+dz
#                 if (nx, ny, nz) in node_map:
#                     diff = (new_img[x,y,z] - new_img[nx,ny,nz]) ** 2
#                     w = np.exp(-diff / (2 * sigma ** 2))
#                     graph.add_edge(idx, node_map[(nx,ny,nz)], w, w)

#         graph.maxflow()
#         for (x, y, z), idx in node_map.items():
#             seg[x, y, z] = graph.get_segment(idx)

#     return seg



import numpy as np
import gc
from time import time
import tracemalloc
from skimage.transform import resize
from scipy.ndimage import binary_erosion, binary_dilation
import maxflow  # assuming PyMaxflow is installed
import matplotlib.pyplot as plt

# ----------------------------
# 1. Coarsening functions for 3D volumes
# ----------------------------

def coarsen(img, factor=2):
    """Coarsen 3D image with padding if needed along all dimensions."""
    d, h, w = img.shape[:3]
    
    # Pad if dimensions aren't divisible by factor
    pad_d = (factor - d % factor) % factor
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    
    # Reshape and apply downsampling along all dimensions
    d_p, h_p, w_p = img.shape[:3]
    img = img.reshape(d_p//factor, factor, h_p//factor, factor, w_p//factor, factor, -1)
    img = img.mean(axis=(1, 3, 5))[:, :, :, 0]
    
    return img[:d//factor, :h//factor, :w//factor]

def coarsen_seeds(seeds, factor=2):
    """
    Coarsen a 3D binary seed volume using max pooling.
    Padding is added to handle dimensions that are not divisible by the factor.
    """
    d, h, w = seeds.shape
    pad_d = (factor - d % factor) % factor
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    padded = np.pad(seeds, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    d_p, h_p, w_p = padded.shape
    # Reshape into blocks of shape (factor, factor, factor) and take maximum
    coarse = padded.reshape(d_p // factor, factor,
                            h_p // factor, factor,
                            w_p // factor, factor).max(axis=(1, 3, 5)).astype(bool)
    return coarse[:d // factor, :h // factor, :w // factor]


# ----------------------------
# 2. Graph Construction in 3D
# ----------------------------


def create_memory_optimized_graph(volume, level, band_width, prev_seg, sigma):
    """
    Memory-efficient graph construction for a 3D volume.
    prev_seg is a 3D segmentation (binary) array.
    """
    graph = maxflow.GraphFloat()
    
    # Pad the segmentation in 3 dimensions
    padded_seg = np.pad(prev_seg,
                        ((band_width, band_width),
                         (band_width, band_width),
                         (band_width, band_width)),
                        mode='reflect')
    
    # Compute inner and outer edges (3D binary morphology)
    inner_edge = binary_erosion(padded_seg, iterations=band_width)
    outer_edge = binary_dilation(padded_seg, iterations=band_width)
    del padded_seg

    # Define sink and source regions (use dilation/erosion in 3D)
    sink_region = binary_dilation(outer_edge) & ~outer_edge
    source_region = ~binary_erosion(inner_edge) & inner_edge

    # Remove padding: note that we assume the padding width is band_width on each side.
    slice_obj = (slice(band_width, -band_width),
                 slice(band_width, -band_width),
                 slice(band_width, -band_width))
    
    sink_pixels = np.argwhere(sink_region[slice_obj])
    source_pixels = np.argwhere(source_region[slice_obj])

    band_mask = (outer_edge ^ inner_edge)[slice_obj]
    del outer_edge, inner_edge

    # Include any pixel that is in the band, source or sink
    band_pixels = np.argwhere(band_mask | source_region[slice_obj] | sink_region[slice_obj])
    del source_region, sink_region, band_mask, slice_obj

    graph.add_nodes(len(band_pixels))
    
    # Create a map from 3D coordinate to node id
    node_map = {tuple(coord): idx for idx, coord in enumerate(band_pixels)}
    
    # Offsets for 6-connected neighborhood in 3D:
    offsets = [(-1, 0, 0), (1, 0, 0),
               (0, -1, 0), (0, 1, 0),
               (0, 0, -1), (0, 0, 1)]
    
    chunk_size = 10000
    for i in range(0, len(band_pixels), chunk_size):
        chunk = band_pixels[i:i+chunk_size]
        
        # Add t-links based on whether the voxel is in sink or source regions.
        for z, y, x in chunk:
            coord = (z, y, x)
            if np.any(np.all(sink_pixels == np.array(coord), axis=1)):
                graph.add_tedge(node_map[coord], 1e9, 0)
            elif np.any(np.all(source_pixels == np.array(coord), axis=1)):
                graph.add_tedge(node_map[coord], 0, 1e9)
            else:
                graph.add_tedge(node_map[coord], 0, 0)
        
        # Add n-links based on differences in the data.
        for z, y, x in chunk:
            for dz, dy, dx in offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                neighbor = (nz, ny, nx)
                if neighbor in node_map:
                    # Compute squared difference. For multi-channel data, adjust accordingly.
                    diff = np.sum((volume[z, y, x] - volume[nz, ny, nx])**2)
                    weight = np.exp(-diff / (2 * sigma**2))
                    graph.add_edge(node_map[(z, y, x)], node_map[neighbor], weight, weight)
    
    return graph, node_map

# ----------------------------
# 3. Multilevel Banded Cuts in 3D
# ----------------------------


def regular_graph_cuts_3d(volume, fg_seeds, bg_seeds, sigma=0.1):
    """
    Baseline full graph cuts for a 3D volume segmentation.
    
    Parameters:
      volume   : 3D image volume (or 3D volume with additional channels, but the first three dims should be spatial)
      fg_seeds : 3D boolean array (shape: (d, h, w)) for foreground seeds
      bg_seeds : 3D boolean array (shape: (d, h, w)) for background seeds
      sigma    : parameter controlling the weight computation
      
    Returns:
      A boolean 3D array representing the segmentation (True for one label, False for the other)
    """
    print("DEBUG SHAPES:")
    print("volume.shape:", volume.shape)
    print("fg_seeds.shape:", fg_seeds.shape)
    print("bg_seeds.shape:", bg_seeds.shape)

    d, h, w = volume.shape[:3]
    
    graph = maxflow.GraphFloat()
    
    # Create a grid of nodes in 3D
    nodeids = graph.add_grid_nodes((d, h, w))
    
    print("Creating the Graph...")

    # Set t-links for all voxels based on seeds
    for z in range(d):
        for y in range(h):
            for x in range(w):
                if fg_seeds[z, y, x]:
                    graph.add_tedge(nodeids[z, y, x], 0, 1e9)
                elif bg_seeds[z, y, x]:
                    graph.add_tedge(nodeids[z, y, x], 1e9, 0)
                # Else, no terminal preference is added (equivalent to 0, 0)
    
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    
    # Add n-links (neighboring connections)
    for z in range(d):
        for y in range(h):
            for x in range(w):
                for dz, dy, dx in offsets:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w:
                        # Compute the squared difference between voxel intensities.
                        # This works for both scalar and multi-channel volumes.
                        diff = np.sum((volume[z, y, x] - volume[nz, ny, nx]) ** 2)
                        weight = np.exp(-diff / (2 * sigma**2))
                        graph.add_edge(nodeids[z, y, x], nodeids[nz, ny, nx], weight, weight)
    
    print("Doing max flow...")
    graph.maxflow()
    segmentation = graph.get_grid_segments(nodeids)
    return segmentation



def memory_optimized_banded_cuts(volume, fg_seeds, bg_seeds, levels=3,
                                 band_width=2, factor=2, sigma=0.1,
                                 compute_baseline=False):
    """
    Perform multilevel graph cuts on a 3D scan.
    
    volume  : 3D image volume (or 3D+channels array)
    fg_seeds: 3D boolean array for foreground seeds
    bg_seeds: 3D boolean array for background seeds
    """
    # Initialize results storage
    all_results = []
    
    # 0. Optionally compute a baseline (full graph cut on full resolution)
    if compute_baseline:
        print("Computing baseline GC...")
        gc.collect()
        tracemalloc.start()
        t0 = time()
        regular_seg = regular_graph_cuts_3d(volume, fg_seeds, bg_seeds, sigma)
        t1 = time()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        all_results.append({
            'type': 'baseline',
            'time': t1 - t0,
            'memory': peak_mem / 1e6,
            'volume': volume.copy(),
            'segmentation': regular_seg.copy()
        })
    else:
        regular_seg = None

    # 1. Build pyramid of coarsened volumes and seeds
    current_vol, current_fg, current_bg = volume, fg_seeds, bg_seeds
    all_results.append({
        'type': 'pyramid',
        'volume': volume.copy(),
        'fg_seeds': fg_seeds,
        'bg_seeds': bg_seeds
    })
    
    pyramids = [(volume, fg_seeds, bg_seeds)]
    for level in range(levels):
        new_vol = coarsen(pyramids[-1][0], factor)
        new_fg = coarsen_seeds(pyramids[-1][1], factor)
        new_bg = coarsen_seeds(pyramids[-1][2], factor)
        pyramids.append((new_vol, new_fg, new_bg))
        all_results.append({
            'type': 'pyramid',
            'volume': new_vol.copy(),
            'fg_seeds': new_fg.copy(),
            'bg_seeds': new_bg.copy()
        })
        del current_vol, current_fg, current_bg
        current_vol, current_fg, current_bg = new_vol, new_fg, new_bg

    # 2. Process the coarsest level with a regular graph cut.
    gc.collect()
    tracemalloc.start()
    t0 = time()
    seg = regular_graph_cuts_3d(current_vol, current_fg, current_bg, sigma)
    t1 = time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    all_results.append({
        'level': levels - 1,
        'type': 'segmentation',
        'time': t1 - t0,
        'memory': peak_mem / 1e6,
        'volume': current_vol.copy(),
        'segmentation': seg.copy()
    })
    
    # 3. Uncoarsen with band refinement
    for level in reversed(range(levels)):
        current_seg = seg.copy()
        t0 = time()
        
        # Project segmentation up to the current level volume
        curr_vol = pyramids[level][0]
        # Here we replace cv2.resize with skimage.transform.resize for 3D volumes.
        seg = resize(current_seg.astype(float), curr_vol.shape[:3], order=0,
                     mode='reflect', preserve_range=True) > 0.5
        
        # Create a band mask using binary dilation and erosion (in 3D)
        band_mask = (binary_dilation(seg, iterations=band_width // 2) ^
                     binary_erosion(seg, iterations=band_width // 2 + 1))
        
        # Build and solve the banded graph
        gc.collect()
        tracemalloc.start()
        print(f"Creating the Graph at level {level}...")
        graph, node_map = create_memory_optimized_graph(curr_vol, level, band_width, seg, sigma)
        print("Computing max flow...")
        graph.maxflow()
        
        # Update segmentation from graph solution
        for coord, nodeid in node_map.items():
            seg[coord] = bool(graph.get_segment(nodeid))
        
        t1 = time()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        all_results.append({
            'level': level,
            'type': 'refinement',
            'time': t1 - t0,
            'memory': peak_mem / 1e6,
            'volume': curr_vol.copy(),
            'segmentation': seg.copy(),
            'band_mask': band_mask.copy()
        })
    
    return regular_seg, seg, all_results

# ============ VISUALISATION =================

def create_detailed_visualization(results, cut, factor, aspect_ratio):
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
        seg_vis = np.zeros((*b['segmentation'].shape[:2], 4))
        seg_vis[b['segmentation'][:, :, cut]] = [0.8, 0.4, 0, 0.4]  # Yellow FG
        seg_vis[~b['segmentation'][:, :, cut]] = [0, 0, 0.8, 0.4]   # Blue BG
        seg_vis[end_seeds['fg_seeds'][:, :, cut]] = np.array([1, 1, 0, 1])  # Yellow
        seg_vis[end_seeds['bg_seeds'][:, :, cut]] = np.array([0, 0, 1, 1])  # Blue
        axs[0,2].imshow(b['volume'][:, :, cut], cmap='gray')
        axs[0,2].imshow(seg_vis)
        axs[0,2].set_title(f"Baseline\nTime spent: {b['time']:.4g}s\nMemory: {b['memory']:.3g}MB")
    # else:
    #     axs[0,2].axis('off')

    # Plot initial seeds
    initial = pyramids[0]
    img_vis = np.stack([initial['volume'][:, :, cut]]*3, axis=-1)
    img_vis = img_vis.copy()
    img_vis = np.concatenate([img_vis, np.ones(img_vis.shape[:2])[:, :, None]], axis=2)
    img_vis[initial['fg_seeds'][:, :, cut]] = [1, 1, 0, 1]  # Yellow
    img_vis[initial['bg_seeds'][:, :, cut]] = [0, 0, 1, 1]  # Blue
    axs[0,0].imshow(img_vis)
    axs[0,0].set_title("Initial Seeds")

    # Plot initial seeds
    end_banded = seg_steps[-1]
    end_seeds = pyramids[0]
    seg_vis = np.zeros((*end_banded['segmentation'].shape[:2], 4))
    seg_vis[end_banded['segmentation'][:, :, cut]] = [0.8, 0.4, 0, 0.4]  # Yellow FG
    seg_vis[~end_banded['segmentation'][:, :, cut]] = [0, 0, 0.8, 0.4]   # Blue BG
    seg_vis[end_seeds['fg_seeds'][:, :, cut]] = [1, 1, 0, 1]  # Yellow
    seg_vis[end_seeds['bg_seeds'][:, :, cut]] = [0, 0, 1, 1]  # Blue
    axs[0,1].imshow(end_banded['volume'][:, :, cut], cmap='gray')
    axs[0,1].imshow(seg_vis)
    axs[0,1].set_title(f"End Banded GC\nTime spent: {sum([s['time'] for s in seg_steps]):.4g}s\nMemory: {max([s['memory'] for s in seg_steps]):.2g}MB")

    for k in range(2, n_cols):
        axs[0,k].axis("off")
    
    # Plot coarse levels and their segmentations
    for i, seg in enumerate(seg_steps, start=0):
        if 'segmentation' in seg:            
            # Overlay segmentation on original image
            corresponding_seed = pyramids[-1 - i]

            seg_vis = np.zeros((*seg['segmentation'].shape[:2], 4))
            seg_vis[seg['segmentation'][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]] = [0.8, 0.4, 0, 0.4]  # Yellow FG
            seg_vis[~seg['segmentation'][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]] = [0, 0, 0.8, 0.4]   # Blue BG

            img_vis = np.stack([corresponding_seed['volume'][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]]*3, axis=-1)
            img_vis = img_vis.copy()
            img_vis = np.concatenate([img_vis, np.ones(img_vis.shape[:2])[:, :, None]], axis=2)

            # Add timing and memory info
            info = f"Time: {seg['time']:.2f}s\nMem: {seg['memory']:.1f}MB"
            if 'band_mask' in seg:
                info += "\n(Banded)"
                seg_vis[seg["band_mask"][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]] = [0, 0, 0, 0.2]
                # axs[1,i].imshow(seg['band_mask'], cmap='gray' if len(seg['band_mask'].shape) == 2 else None)
            
            seg_vis[corresponding_seed['fg_seeds'][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]] = [1, 1, 0, 1]  # Yellow
            seg_vis[corresponding_seed['bg_seeds'][:, :, int(cut/(factor**(len(seg_steps)-i-1)))]] = [0, 0, 1, 1]  # Blue
            
            axs[1,i].imshow(img_vis)
            axs[1,i].imshow(seg_vis)
            axs[1, i].set_title(f"Coarse {n_cols-1-i} - {seg['time']:.2g}s - {seg['memory']:.2g}MB")
            # axs[1,i].imshow(seg_vis)
    # Hide unused axes
    for j in range(0, n_cols):
        axs[0,j].axis('off')
        axs[1,j].axis('off')
    
    
    plt.suptitle("Banded Graph Cuts Progression")
    # fig.savefig(os.path.join(os.path.dirname(IMAGE_PATH), "results", os.path.basename(IMAGE_PATH),"all_res.png"))
    plt.show()



baseline_seg, banded_seg, all_results = memory_optimized_banded_cuts(volume,fg_3d,bg_3d,levels=3,band_width=2,factor=2,sigma=0.01, compute_baseline=False)

cut = slice_idx

create_detailed_visualization(all_results, cut, factor=2, aspect_ratio=volume.shape[1]/volume.shape[0])


import SimpleITK as sitk

banded_seg = banded_seg.astype(int)
sitk_arr = sitk.GetImageFromArray(banded_seg)
sitk_arr.SetSpacing((1.0, 1.0, 1.0))
sitk.WriteImage(sitk_arr, VOXEL_PATH.replace("scan.nii", "seg.mha"))

baseline_seg = baseline_seg.astype(int)
sitk_arr = sitk.GetImageFromArray(baseline_seg)
sitk_arr.SetSpacing((1.0, 1.0, 1.0))
sitk.WriteImage(sitk_arr, VOXEL_PATH.replace("scan.nii", "base_seg.mha"))

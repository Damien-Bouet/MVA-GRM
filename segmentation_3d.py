import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_erosion

FG_PATH = "/home/alex/Modèles/slice_120_seeds/foreground.npy"
BG_PATH = "/home/alex/Modèles/slice_120_seeds/background.npy"
VOXEL_PATH = "/home/alex/Téléchargements/image1/liver_1.nii"


nii = nib.load(VOXEL_PATH)

volume = nii.get_fdata() #np array


#graph cuts
def regular_graph_cuts_3d(volume, fg_seeds, bg_seeds, sigma=0.1):
    print("DEBUG SHAPES:")
    print("volume.shape:", volume.shape)
    print("fg_seeds.shape:", fg_seeds.shape)
    print("bg_seeds.shape:", bg_seeds.shape)

    assert volume.shape == fg_seeds.shape == bg_seeds.shape, "Shape mismatch!"
    
    D, H, W = volume.shape
    graph = maxflow.GraphFloat()
    nodeids = graph.add_grid_nodes((D, H, W))
    print("Creating the Graph...")

    # Add t-links
    for z in range(D):
        for y in range(H):
            for x in range(W):
                if fg_seeds[z, y, x]:
                    graph.add_tedge(nodeids[z, y, x], 0, 1e9)
                elif bg_seeds[z, y, x]:
                    graph.add_tedge(nodeids[z, y, x], 1e9, 0)

    # 6-connected neighborhood
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for z in range(D):
        for y in range(H):
            for x in range(W):
                for dz, dy, dx in offsets:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                        diff = (volume[z, y, x] - volume[nz, ny, nx]) ** 2
                        weight = np.exp(-diff / (2 * sigma ** 2))
                        graph.add_edge(nodeids[z, y, x], nodeids[nz, ny, nx], weight, weight)
    print("doing maxflow")
    graph.maxflow()
    return graph.get_grid_segments(nodeids)

def expand_foreground_seed(fg_3d, radius=1):
    # Create spherical structuring element
    zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 + zz**2 <= radius**2

    fg_dilated = binary_dilation(fg_3d, structure=mask)
    return fg_dilated




#experiment

#3d segmentation

slice_idx = 80  
fg_2d = np.load(FG_PATH)  # shape: (H, W)
bg_2d = np.load(BG_PATH)

z_start = 30
z_end = 90  # just 20 slices instead of full depth

H, W, D = volume.shape

subvol = volume[:, :, z_start:z_end]

fg_3d = np.zeros(volume.shape, dtype=bool)
bg_3d = np.zeros(volume.shape, dtype=bool)
fg_3d[:, :, slice_idx] = fg_2d
bg_3d[:, :, slice_idx] = bg_2d

fg_expanded = expand_foreground_seed(fg_3d, radius=1)

seg_sub = regular_graph_cuts_3d(subvol, fg_expanded[:, :, z_start:z_end], bg_3d[:, :, z_start:z_end], sigma=5)

np.save("regular_segmentation_3d_mask.npy", seg_sub.astype(np.uint8))

#3d banded cut

def downsample_3d(volume, factor=2, is_mask=False):
    """Downsample a 3D volume along XY using cv2, slice by slice"""
    depth = volume.shape[2]
    slices = []
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    for i in range(depth):
        resized = cv2.resize(volume[:, :, i], 
                             (volume.shape[1] // factor, volume.shape[0] // factor), 
                             interpolation=interp)
        slices.append(resized)

    return np.stack(slices, axis=2)


def banded_cuts_3d(volume, fg_seeds, bg_seeds, levels=2, band_width=2, factor=2, sigma=0.1):
    all_results = []
    pyramids = [(volume.copy(), fg_seeds.copy(), bg_seeds.copy())]

    # 1. Coarsen volume & seeds
    for level in range(levels):
        v, f, b = pyramids[-1]
        v_c = downsample_3d(v, factor=factor)
        f_c = downsample_3d(f.astype(np.uint8), factor=factor, is_mask=True).astype(bool)
        b_c = downsample_3d(b.astype(np.uint8), factor=factor, is_mask=True).astype(bool)
        pyramids.append((v_c, f_c, b_c))

    # 2. Solve coarsest
    current_img, current_fg, current_bg = pyramids[-1]
    #seg = regular_graph_cuts_3d(current_img, current_fg, current_bg, sigma)
    seg = regular_graph_cuts_3d(np.transpose(current_img, (2, 0, 1)),np.transpose(current_fg, (2, 0, 1)),np.transpose(current_bg, (2, 0, 1)),sigma)
    # (D, H, W) apparemment convention différente

    # 3. Uncoarsening loop
    for level in reversed(range(levels)):
        prev_seg = seg.astype(np.uint8)
        new_img, _, _ = pyramids[level]
        # Resize segmentation to next finer level
        depth = new_img.shape[2]
        resized = []
        for i in range(depth):
            resized_slice = cv2.resize(prev_seg[:, :, i], 
                                       (new_img.shape[1], new_img.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            resized.append(resized_slice)
        seg = np.stack(resized, axis=2).astype(bool)

        band_mask = binary_dilation(seg, iterations=band_width) ^ binary_erosion(seg, iterations=band_width)
        band_voxels = np.argwhere(band_mask)

        graph = maxflow.GraphFloat()
        node_map = {}
        nodeids = graph.add_nodes(len(band_voxels))
        for idx, (x, y, z) in enumerate(band_voxels):
            node_map[(x, y, z)] = idx
            # T-links
            if fg_seeds[x, y, z]:
                graph.add_tedge(idx, 0, 1e9)
            elif bg_seeds[x, y, z]:
                graph.add_tedge(idx, 1e9, 0)
            else:
                graph.add_tedge(idx, 0, 0)

        # N-links
        offsets = [(1,0,0), (0,1,0), (0,0,1)]
        for (x, y, z), idx in node_map.items():
            for dx, dy, dz in offsets:
                nx, ny, nz = x+dx, y+dy, z+dz
                if (nx, ny, nz) in node_map:
                    diff = (new_img[x,y,z] - new_img[nx,ny,nz]) ** 2
                    w = np.exp(-diff / (2 * sigma ** 2))
                    graph.add_edge(idx, node_map[(nx,ny,nz)], w, w)

        graph.maxflow()
        for (x, y, z), idx in node_map.items():
            seg[x, y, z] = graph.get_segment(idx)

    return seg



segmentation_3d = memory_optimized_banded_cuts_3d(volume,fg_3d,bg_3d,levels=2,band_width=2,factor=2,sigma=10)

np.save("segmentation_3d_mask.npy", segmentation_3d.astype(np.uint8))

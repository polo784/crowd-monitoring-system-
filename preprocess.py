import numpy as np
import scipy.io as sio
import os
import h5py

def generate_density_map(img_shape, points, sigma=15):
    """
    Translates discrete x,y coordinates into a continuous spatial density matrix.
    img_shape: (height, width) of the Mall Dataset images (480, 640)
    points: Array of [x, y] coordinates
    sigma: Spread of the Gaussian blur. 15 is standard for crowded scenes.
    """
    density_map = np.zeros(img_shape, dtype=np.float32)

    if len(points) == 0:
        return density_map

    # Match scipy.ndimage.gaussian_filter's default 4-sigma kernel radius.
    radius = int(4 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian_1d = np.exp(-(offsets**2) / (2 * sigma**2))
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)

    for p in points:
        # The Mall dataset sometimes stores floats, so we cast to int
        x, y = int(p[0]), int(p[1])
        # Ensure coordinates are strictly within matrix bounds
        if y < img_shape[0] and x < img_shape[1] and y >= 0 and x >= 0:
            image_y_start = max(0, y - radius)
            image_y_end = min(img_shape[0], y + radius + 1)
            image_x_start = max(0, x - radius)
            image_x_end = min(img_shape[1], x + radius + 1)

            kernel_y_start = image_y_start - (y - radius)
            kernel_y_end = kernel_y_start + (image_y_end - image_y_start)
            kernel_x_start = image_x_start - (x - radius)
            kernel_x_end = kernel_x_start + (image_x_end - image_x_start)

            visible_kernel = gaussian_kernel[
                kernel_y_start:kernel_y_end,
                kernel_x_start:kernel_x_end,
            ]
            visible_kernel = visible_kernel / visible_kernel.sum()

            # Add rather than assign so annotations at the same pixel accumulate.
            density_map[
                image_y_start:image_y_end,
                image_x_start:image_x_end,
            ] += visible_kernel

    return density_map

def process_mall_dataset(mat_path, output_dir, img_shape=(480, 640)):
    """
    Reads the Mall Dataset MATLAB file and exports target matrices as .h5 files.
    """
    print(f"Creating output directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading MATLAB coordinate data from {mat_path}...")
    try:
        mat_data = sio.loadmat(mat_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {mat_path}. Check your folder structure!")
        return
    
    # The Mall dataset stores coordinates in a specific nested MATLAB structure
    # mat_data['frame'][0] contains an array where each index is a frame
    frames = mat_data['frame'][0]
    total_frames = len(frames)
    print(f"Found {total_frames} frames to process. Generating heatmaps...")
    
    for i, frame in enumerate(frames):
        # Extract the x,y coordinates for the current image. 
        # If there are no people, the shape might be weird, so we handle it gracefully.
        try:
            points = frame['loc'][0][0]
        except IndexError:
            points = []
            
        # Generate the mathematical heatmap
        density_matrix = generate_density_map(img_shape, points)
        
        # We name the files to match the image names: seq_000001 -> frame_0001.h5
        # This makes it super easy to pair them up during training!
        h5_path = os.path.join(output_dir, f"frame_{i+1:04d}.h5")
        
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('density', data=density_matrix)
            
        if (i + 1) % 200 == 0 or (i + 1) == total_frames:
            print(f"Processed {i + 1} / {total_frames} matrices...")
            
    print("\n✅ Transformation complete! Target matrices are ready in the density_maps folder.")

if __name__ == "__main__":
    # Pointing directly to your newly renamed folder
    mat_file_path = 'cv_project/mall_dataset/mall_gt.mat' 
    output_folder = 'cv_project/mall_dataset/density_maps'
    
    process_mall_dataset(mat_file_path, output_folder)

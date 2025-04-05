# create_pic6_and_3d.py
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import os
import pickle

def get_resault(weighted_sum, threshold, grayscale_array, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Convert threshold to grayscale (fray) scale.
    thresh_value = threshold * 255
    # Create a binary image: 1 where weighted_sum > threshold, else 0.
    binary_array = (weighted_sum > threshold).astype(np.uint8)

    # Create a mask for pixels that are originally > 0 in weighted_sum.
    nzero_mask_stack = np.count_nonzero(weighted_sum.astype(np.float64))
    nzero_mask_threshold = np.count_nonzero(binary_array.astype(np.float64))
    lost_pixles = (nzero_mask_stack-nzero_mask_threshold)
    present = lost_pixles*100.0/nzero_mask_stack
    # return present

    # print(nzero_mask_stack,nzero_mask_threshold,present)
    img = Image.fromarray(binary_array * 255, mode='L')
    threshold_name = round(threshold*100,1)
    img.save(os.path.join(output_dir,f"{threshold_name}% threshold.png"))
    print(f"{threshold*100}% thresholded image has been created and saved as '{threshold*100}% threshold.png'.")

    # --- 3D Visualization of pic5 with a Red Threshold Sheet ---
    # Create coordinate grids for the image.
    width, height = img.size
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y)
    Z = grayscale_array  # Grayscale intensities in 0-255
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface of pic5 using a gray colormap.
    surf = ax.plot_surface(X, Y, Z, cmap='gray', linewidth=0, antialiased=False)

    # Create a constant "sheet" at z = thresh_value.
    Zsheet = np.full_like(X, thresh_value)
    # Plot the red sheet with some transparency.
    sheet = ax.plot_surface(X, Y, Zsheet, color='red', alpha=0.5)

    ax.set_xlabel('X (Image Column)')
    ax.set_ylabel('Y (Image Row)')
    ax.set_zlabel('Grayscale Value (Z)')
    ax.set_title('3D Visualization of Stacking with Red Threshold Sheet')
    pickle_filename = f"{threshold_name}%threshold.pkl"
    pickle_path = os.path.join(output_dir, pickle_filename)
    with open(pickle_path, "wb") as f:
     pickle.dump(fig, f)
    # print(f"Figure saved as '{pickle_path}'.")
    
    # Save the 3D plot with threshold sheet.
    file_name = (f"{threshold_name}%.png")
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()
    # print(f"3D plot with threshold sheet has been saved as {file_path}.")
    #plt.show()
    return present
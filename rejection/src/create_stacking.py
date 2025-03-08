from PIL import Image
import numpy as np
import os

def stacking(model_images, grades, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Convert images to numpy arrays and normalize to [0,1]
    arrays = []
    for img in model_images:
        arrays.append(np.array(img, dtype=float) / 255.0)

    weighted_sum=0
    weighted = 0
    # Process each image by multiplying with its grade and save as noise versions.
    for i in range(4):
        weighted = arrays[i] * grades[i]
        grayscale_array = (weighted * 255).astype(np.uint8)
        noise_img = Image.fromarray(grayscale_array, mode='L')
        noise_img.save(os.path.join(output_dir, f"mask_model_{i+1}+noise.png"))
        weighted_sum = weighted_sum + weighted
    weighted_sum = weighted_sum/4

    # --- Create pic5 as the weighted average of the four images ---
    # weighted_sum = (arrays[0] * grades[0] + arrays[1] * grades[1] +
    #                 arrays[2] * grades[2] + arrays[3] * grades[3]) / 4.0
    grayscale_array = (weighted_sum * 255).astype(np.uint8)
    stack = Image.fromarray(grayscale_array, mode='L')
    stack.save(os.path.join(output_dir, "stack.png"))
    print("Stack has been created and saved as 'stack.png'.")
    print("Max grayscale value in stack:", grayscale_array.max())
    return stack, weighted_sum, grayscale_array

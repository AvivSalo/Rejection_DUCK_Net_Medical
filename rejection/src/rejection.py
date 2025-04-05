import os
import pandas as pd
from PIL import Image
from create_stacking import stacking
from resault import get_resault
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm  # Import normal distribution
import shutil

global_array = np.zeros((100, 41), dtype=float)

def check_rej():
    global global_array
    # Path to the CSV file (update with your file's path)
    csv_file = "data/scores.csv"
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Loop over each row in the CSV
    for idx, row in df.iterrows():
        image_name = row["Image Name"]  # e.g., "polip_1"
        image_name = image_name.replace(".jpg", "")
        scores = [row["Score_1"], row["Score_2"], row["Score_3"], row["Score_4"]]
        # Construct the folder path where the images are stored (e.g., "data/polip_1")
        folder_path = os.path.join("data", image_name)
        # List to hold the model output images
        model_images = []
        
        # Read pictures pic1.png to pic4.png
        for i in range(1, 5):
            image_file = os.path.join(folder_path, f"mask_model_{i}.png")
            if os.path.exists(image_file):
                img = Image.open(image_file).convert("L")
                model_images.append(img)
            else:
                print(f"Warning: File {image_file} not found.")
        
        # Now you have the images and scores for the given image name.
        # For example, you can print out the scores and image sizes:
        # print(f"Image Name: {image_name}")
        # print(f"Scores: {scores}")
        stack, weighted_sum, grayscale_array = stacking(model_images, scores, folder_path)
        model_images.append(stack)
        # Create thresholds from 0 to 0.8 in steps of 0.1, and from 0.8 to 1 in steps of 0.01
        thresholds_part1 = np.arange(0, 1.01, 0.025)
        cnt=0
        for threshold in thresholds_part1:
            rounded_number = round(threshold, 3)
            folder_label = f"{rounded_number*100:.2f}%"
            new_folder_name = f"{folder_label}_threshold"
            new_folder_path = os.path.join(folder_path, new_folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Folder '{new_folder_path}' created.")
            else:
                print(f"Folder '{new_folder_path}' already exists.")
            global_array [int(image_name)][cnt]= get_resault(weighted_sum, rounded_number, grayscale_array, new_folder_path)
            cnt+=1
        os.system('cls' if os.name == 'nt' else 'clear')
                

    # for i in range(global_array.shape[1]):
    #     matplotlib.use("Agg")
    #     plt.figure(figsize=(10, 5))

    #     # Compute statistics
    #     mean_value = np.mean(global_array[:, i])  # Mean of the data
    #     std_dev = np.std(global_array[:, i])  # Standard deviation

    #     # Generate values for the normal distribution curve
    #     x_values = np.linspace(min(global_array[:, i]), max(global_array[:, i]), 100)
    #     normal_curve = norm.pdf(x_values, mean_value, std_dev) * len(global_array[:, i]) * 10  # Scale it for visibility

    #     # Create a bar graph
    #     plt.bar(range(100), global_array[:, i], color="blue", alpha=0.6, label="Filtered Pixels")

    #     # Overlay normal distribution curve
    #     plt.plot(x_values, normal_curve, color='red', linestyle='--', linewidth=2, label="Normal Distribution")

    #     # # Add statistics as text annotation
    #     # plt.text(0.5, max(global_array[:, i]) * 0.95, f"Mean: {mean_value:.2f}\nStd Dev: {std_dev:.2f}",
    #     #         fontsize=12, color="black", bbox=dict(facecolor="white", alpha=0.5))

    #     # Labels and title
    #     plt.xlabel("Picture Index")
    #     plt.ylabel("% of Filtered Pixels")
    #     plt.title(f"Grayscale Threshold Changes for Value {(i) * 25.5}, Mean:{mean_value:.2f},STD Dev{std_dev:.2f}")
    #     plt.legend()
    #     plt.grid(True, axis="y", linestyle="--", alpha=0.7)  # Grid on Y-axis only

    #     # Define the file path
    #     file_path = os.path.join("data/graphs", f"threshold_plot_{(i) * 25.5:.1f}.png")

    #     # Save the figure
    #     plt.savefig(file_path, dpi=300, bbox_inches='tight')

    #     # Close the figure to free memory
    #     plt.close()


##################DO NOT TOUCH#####################################
    thresholds = (np.linspace(0, 255, global_array.shape[1])).astype(int)
     # Converts row indices to real thresholds (11 values)

# Loop over each image (row)
    for i in range(global_array.shape[0]):
        matplotlib.use("Agg")
        plt.figure(figsize=(10, 5))

        # Get data for this image across thresholds
        row_data = global_array[i, :]  # shape = (11,)

        # Compute statistics
        mean_value = np.mean(row_data)
        std_dev = np.std(row_data)

        # Generate smooth normal curve
        x_values = np.linspace(min(thresholds), max(thresholds), 100)
        normal_curve = norm.pdf(x_values, mean_value, std_dev) * len(row_data) * 10

        # Bar plot
        # Set a tighter width so bars don't overlap
        bar_width = 4  # or even try 8 or 6

        plt.bar(thresholds, row_data, width=bar_width, color="blue", alpha=0.6, label="Filtered Pixels", align="center")
        plt.plot(x_values, normal_curve, color='red', linestyle='--', linewidth=2, label="Normal Distribution")

        plt.xlabel("Threshold Value (0–255)")
        plt.ylabel("% of Filtered Pixels")
        plt.title(f"Image {i}: Threshold vs Filtered Pixels\nMean: {mean_value:.2f}, STD: {std_dev:.2f}")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Tighter and cleaner X-ticks
        plt.xticks(np.arange(0, 256, 10))

        # Optional: add tighter X limits if needed
        plt.xlim(-5, 260)

        # Save the figure
        os.makedirs("data/pic", exist_ok=True)
        file_path = os.path.join("data/pic", f"threshold_plot_img_{i}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    target_values = np.arange(0, 101, 2.5) # 0.0 to 1.0 in steps of 0.1
    #tatget_value = 20
    closest_indices = []
    for row in global_array:
        row_results = []
        # idx = np.argmin(np.abs(row - tatget_value))
        # row_results.append(idx)
        for target in target_values:
        # Find the column index where the value is closest to the target
            idx = np.argmin(np.abs(row - target))
            row_results.append(idx)
        closest_indices.append(row_results)
    print("here")
    for row_index, row in enumerate(closest_indices):
        for target_idx, threshold_idx in enumerate(row):
    # Compute the actual threshold value
            threshold_value = threshold_idx
            target_percent = target_values[target_idx]

            # Folder name: 80.00%_threshold
            threshold_folder = f"{threshold_value:.2f}%_threshold"
            # File name: 80.0% threshold.png
            threshold_filename = f"{threshold_value:.1f}% threshold.png"

            # Full source path
            src_path = os.path.join("data", str(row_index), threshold_folder, threshold_filename)

            # Destination
            dst_folder = os.path.join("data", "results", str(target_percent))
            dst_path = os.path.join(dst_folder, f"{row_index}.png")

            # Make sure destination folder exists
            os.makedirs(dst_folder, exist_ok=True)

            try:
                shutil.copy(src_path, dst_path)
            except FileNotFoundError:
                print(f"❌ File not found: {src_path}")
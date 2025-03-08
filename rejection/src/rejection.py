import os
import pandas as pd
from PIL import Image
from create_stacking import stacking
from resault import get_resault
import numpy as np

def check_rej():
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
        print(f"Image Name: {image_name}")
        print(f"Scores: {scores}")
        stack, weighted_sum, grayscale_array = stacking(model_images, scores, folder_path)
        model_images.append(stack)
        for threshold in np.arange(0, 1.01, 0.1):  # 1.01 ensures that 1 is included
            rounded_number = round(threshold, 1)
            new_folder_name = f"{rounded_number*100}%_threshold"
            new_folder_path = os.path.join(folder_path, new_folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Folder '{new_folder_path}' created.")
            else:
                print(f"Folder '{new_folder_path}' already exists.")
            get_resault(weighted_sum, rounded_number,grayscale_array, new_folder_path)
        for i, img in enumerate(model_images, start=1):
            print(f"pic{i}.png size: {img.size}")
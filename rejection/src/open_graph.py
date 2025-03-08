import os
import pickle
import matplotlib.pyplot as plt

pickle_path = r"/Users/liav/Desktop/GIT/duck/data/13/50.0%_threshold/50.0%threshold.pkl"



with open(pickle_path, "rb") as f:
    loaded_fig = pickle.load(f)

    # Register the loaded figure with pyplot so that it can be managed
plt.figure(loaded_fig.number)
loaded_fig.canvas.draw_idle()

    # Display the figure
plt.show(block=True)
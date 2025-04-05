import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation


pickle_path = r"data/92/37.00%_threshold/37.0%threshold.pkl"



with open(pickle_path, "rb") as f:
    loaded_fig = pickle.load(f)

#     # Register the loaded figure with pyplot so that it can be managed
# plt.figure(loaded_fig.number)
# loaded_fig.canvas.draw_idle()

#     # Display the figure
# plt.show(block=True)



# Register the loaded figure with pyplot
plt.figure(loaded_fig.number)
loaded_fig.canvas.draw_idle()

# Retrieve the existing axes (assuming the first is your 3D axis)
ax = loaded_fig.axes[0]

# Define an update function for the animation that rotates the view
def update(angle):
    ax.view_init(azim=angle)  # Update the azimuth angle
    return loaded_fig,

# Create an animation that rotates from 0 to 360 degrees
ani = animation.FuncAnimation(loaded_fig, update, frames=range(0, 360, 2), interval=50, blit=False)

# Display the figure with the animation
plt.show()
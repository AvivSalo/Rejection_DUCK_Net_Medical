from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
import numpy as np
from PIL import Image

def over_lap_metrix(gt_array, predictio_array):
    # Flatten and binarize the arrays
    gt_flat = np.ndarray.flatten(np.array(gt_array, dtype=bool))
    pred_flat = np.ndarray.flatten(np.array(predictio_array) > 0.5)

    # Metrics
    dice = f1_score(gt_flat, pred_flat)
    jaccard = jaccard_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    accuracy = accuracy_score(gt_flat, pred_flat)

    return dice, jaccard, precision, recall, accuracy

gt_images = []
for i in range(0,100):
    gt = Image.open(f"data/masks/{i}.jpg").convert("L")
    gt_images.append(gt)

    
max_dice = max_jac = max_precision = max_recall = max_acc = 0
flag_dice = flag_jaccard = flag_precision = flag_recall = flag_acc = 0

for j in np.arange(0, 101, 2.5):
    model_images = []
    for i in range(0, 100):
        predict = Image.open(f"data/results/{j}/{i}.png").convert("L")
        model_images.append(predict)

    dice, jaccard, precision, recall, accuracy = over_lap_metrix(gt_images, model_images)

    # Update max values and flags
    if dice > max_dice:
        max_dice = dice
        flag_dice = j
    if jaccard > max_jac:
        max_jac = jaccard
        flag_jaccard = j
    if precision > max_precision:
        max_precision = precision
        flag_precision = j
    if recall > max_recall:
        max_recall = recall
        flag_recall = j
    if accuracy > max_acc:
        max_acc = accuracy
        flag_acc = j

    print(f"For TH {j}%:")
    print(f"  Dice:     {dice:.4f}")
    print(f"  Jaccard:  {jaccard:.4f}")
    print(f"  Precision:{precision:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print("")

# Print best results
print(f"Best Dice:     {max_dice:.4f} at {flag_dice}% TH")
print(f"Best Jaccard:  {max_jac:.4f} at {flag_jaccard}% TH")
print(f"Best Precision:{max_precision:.4f} at {flag_precision}% TH")
print(f"Best Recall:   {max_recall:.4f} at {flag_recall}% TH")
print(f"Best Accuracy: {max_acc:.4f} at {flag_acc}% TH")
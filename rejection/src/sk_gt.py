from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
from PIL import Image

def over_lap_metrix(gt_array, predictio_array):
    #dice
    dice_test = f1_score(np.ndarray.flatten(np.array(gt_array, dtype=bool)),
                            np.ndarray.flatten(np.array(predictio_array) > 0.5))

    #mIOU
    jaccard_score_test = jaccard_score(np.ndarray.flatten(np.array(gt_array, dtype=bool)),
                            np.ndarray.flatten(np.array(predictio_array) > 0.5))
    print(dice_test)
    print(jaccard_score_test)

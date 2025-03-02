import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, Callback
from datetime import datetime
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import DUCK_Net
from ImageLoader import ImageLoader2D
import random

# Checking the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow Version:", tf.__version__)

# Setting the model parameters
img_size = 352
dataset_type = 'kvasir'
learning_rate = 1e-4
seed_value = 42
filters = 17
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
BATCH_SIZE = 2
data_path = "/home/aviv.salomon/Documents/MSC_Intelligence_Systems/Semester_A_second_year/Computer_vision/DUCK-Net/Data/Kvasir-SEG/"

ct = datetime.now()
model_type = "DuckNet"

progress_path = 'ProgressFull/' + dataset_type + '_progress_csv_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.csv'
os.makedirs(os.path.dirname(progress_path), exist_ok=True)
progressfull_path = 'ProgressFull/' + dataset_type + '_progress_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.txt'
plot_path = 'ProgressFull/' + dataset_type + '_progress_plot_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.png'
model_save_path_h5 = 'ModelSaveTensorFlow/' + dataset_type + '/' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.h5'
model_save_path_tf = 'ModelSaveTensorFlow/' + dataset_type + '/' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '_tf'
print(f"Model will be saved as .h5 to: {model_save_path_h5}")
print(f"Model will be saved as TF format to: {model_save_path_tf}\n")
os.makedirs(os.path.dirname(model_save_path_h5), exist_ok=True)
os.makedirs(model_save_path_tf, exist_ok=True)

# TensorBoard log directory
tensorboard_log_dir = 'logs/' + dataset_type + '_' + model_type + '_filters_' + str(filters) + '_' + str(ct)
os.makedirs(tensorboard_log_dir, exist_ok=True)

pretrained_model_path = '/home/aviv.salomon/Documents/MSC_Intelligence_Systems/Semester_A_second_year/Computer_vision/DUCK-Net/ModelSaveTensorFlow/DuckNet_17_Kvasir_Tf_Model'  # Path to pre-trained SavedModel folder

EPOCHS = 25
min_loss_for_saving = float('inf')  # Initialize to infinity to save the first valid loss
aug_probability = 1.0

# Loading the data directly from train, validation, and test folders
x_train, y_train = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', data_path, split='train')
x_valid, y_valid = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', data_path, split='validation')
x_test, y_test = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', data_path, split='test')

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Defining the augmentations
aug_train = albu.Compose([
    albu.HorizontalFlip(p=aug_probability),
    albu.VerticalFlip(p=aug_probability),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, p=aug_probability),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), p=aug_probability),
])

def augment_images():
    x_train_out = []
    y_train_out = []
    for i in range(len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])
    return np.array(x_train_out), np.array(y_train_out)

# Custom TensorBoard callback for metrics and images
class CustomTensorBoardCallback(Callback):
    def __init__(self, log_dir, x_train, y_train, x_valid, y_valid, x_test, y_test):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid
        self.x_test, self.y_test = x_test, y_test
        # Select one random image from train and validation
        self.train_idx = random.randint(0, len(x_train) - 1)
        self.valid_idx = random.randint(0, len(x_valid) - 1)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            # Predictions for all sets
            pred_train = self.model.predict(self.x_train, batch_size=BATCH_SIZE, verbose=0)
            pred_valid = self.model.predict(self.x_valid, batch_size=BATCH_SIZE, verbose=0)
            pred_test = self.model.predict(self.x_test, batch_size=BATCH_SIZE, verbose=0)

            # Calculate metrics
            dice_train = f1_score(np.ndarray.flatten(np.array(self.y_train, dtype=bool)), np.ndarray.flatten(pred_train > 0.5))
            dice_valid = f1_score(np.ndarray.flatten(np.array(self.y_valid, dtype=bool)), np.ndarray.flatten(pred_valid > 0.5))
            dice_test = f1_score(np.ndarray.flatten(np.array(self.y_test, dtype=bool)), np.ndarray.flatten(pred_test > 0.5))

            miou_train = jaccard_score(np.ndarray.flatten(np.array(self.y_train, dtype=bool)), np.ndarray.flatten(pred_train > 0.5))
            miou_valid = jaccard_score(np.ndarray.flatten(np.array(self.y_valid, dtype=bool)), np.ndarray.flatten(pred_valid > 0.5))
            miou_test = jaccard_score(np.ndarray.flatten(np.array(self.y_test, dtype=bool)), np.ndarray.flatten(pred_test > 0.5))

            precision_train = precision_score(np.ndarray.flatten(np.array(self.y_train, dtype=bool)), np.ndarray.flatten(pred_train > 0.5))
            precision_valid = precision_score(np.ndarray.flatten(np.array(self.y_valid, dtype=bool)), np.ndarray.flatten(pred_valid > 0.5))
            precision_test = precision_score(np.ndarray.flatten(np.array(self.y_test, dtype=bool)), np.ndarray.flatten(pred_test > 0.5))

            recall_train = recall_score(np.ndarray.flatten(np.array(self.y_train, dtype=bool)), np.ndarray.flatten(pred_train > 0.5))
            recall_valid = recall_score(np.ndarray.flatten(np.array(self.y_valid, dtype=bool)), np.ndarray.flatten(pred_valid > 0.5))
            recall_test = recall_score(np.ndarray.flatten(np.array(self.y_test, dtype=bool)), np.ndarray.flatten(pred_test > 0.5))

            accuracy_train = accuracy_score(np.ndarray.flatten(np.array(self.y_train, dtype=bool)), np.ndarray.flatten(pred_train > 0.5))
            accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(self.y_valid, dtype=bool)), np.ndarray.flatten(pred_valid > 0.5))
            accuracy_test = accuracy_score(np.ndarray.flatten(np.array(self.y_test, dtype=bool)), np.ndarray.flatten(pred_test > 0.5))

            # Log scalar metrics
            tf.summary.scalar('train/loss', logs.get('loss', 0), step=epoch)
            tf.summary.scalar('valid/loss', logs.get('val_loss', 0), step=epoch)
            tf.summary.scalar('test/loss', dice_metric_loss(self.y_test, pred_test).numpy(), step=epoch)

            tf.summary.scalar('train/dice', dice_train, step=epoch)
            tf.summary.scalar('valid/dice', dice_valid, step=epoch)
            tf.summary.scalar('test/dice', dice_test, step=epoch)

            tf.summary.scalar('train/miou', miou_train, step=epoch)
            tf.summary.scalar('valid/miou', miou_valid, step=epoch)
            tf.summary.scalar('test/miou', miou_test, step=epoch)

            tf.summary.scalar('train/precision', precision_train, step=epoch)
            tf.summary.scalar('valid/precision', precision_valid, step=epoch)
            tf.summary.scalar('test/precision', precision_test, step=epoch)

            tf.summary.scalar('train/recall', recall_train, step=epoch)
            tf.summary.scalar('valid/recall', recall_valid, step=epoch)
            tf.summary.scalar('test/recall', recall_test, step=epoch)

            tf.summary.scalar('train/accuracy', accuracy_train, step=epoch)
            tf.summary.scalar('valid/accuracy', accuracy_valid, step=epoch)
            tf.summary.scalar('test/accuracy', accuracy_test, step=epoch)

            # Log one random image from train and validation
            train_image = self.x_train[self.train_idx:self.train_idx+1]
            train_mask = self.y_train[self.train_idx:self.train_idx+1]
            train_pred = self.model.predict(train_image, batch_size=1, verbose=0)
            valid_image = self.x_valid[self.valid_idx:self.valid_idx+1]
            valid_mask = self.y_valid[self.valid_idx:self.valid_idx+1]
            valid_pred = self.model.predict(valid_image, batch_size=1, verbose=0)

            # Ensure 4D and 3 channels
            if len(train_mask.shape) == 3:
                train_mask = np.expand_dims(train_mask, axis=-1)
            if len(train_pred.shape) == 3:
                train_pred = np.expand_dims(train_pred, axis=-1)
            if len(valid_mask.shape) == 3:
                valid_mask = np.expand_dims(valid_mask, axis=-1)
            if len(valid_pred.shape) == 3:
                valid_pred = np.expand_dims(valid_pred, axis=-1)

            train_mask = np.repeat(train_mask, 3, axis=-1)
            train_pred = np.repeat(train_pred, 3, axis=-1)
            valid_mask = np.repeat(valid_mask, 3, axis=-1)
            valid_pred = np.repeat(valid_pred, 3, axis=-1)

            train_images = np.concatenate([train_image, train_mask, train_pred], axis=2)
            valid_images = np.concatenate([valid_image, valid_mask, valid_pred], axis=2)

            tf.summary.image('train/image_mask_pred', train_images, step=epoch, max_outputs=1)
            tf.summary.image('valid/image_mask_pred', valid_images, step=epoch, max_outputs=1)

        self.writer.flush()

# Loading and compiling the pre-trained model
print(f"Loading pre-trained model from: {pretrained_model_path}")
if not os.path.exists(pretrained_model_path) or not os.path.isfile(os.path.join(pretrained_model_path, 'saved_model.pb')):
    raise FileNotFoundError(f"Pre-trained model directory not found or invalid at {pretrained_model_path}. Ensure it contains 'saved_model.pb' and 'variables/' subfolder.")
model = tf.keras.models.load_model(pretrained_model_path, custom_objects={'dice_metric_loss': dice_metric_loss})
model.compile(optimizer=optimizer, loss=dice_metric_loss)

model.summary()

# Training the model
step = 0

tensorboard_callback = CustomTensorBoardCallback(tensorboard_log_dir, x_train, y_train, x_valid, y_valid, x_test, y_test)

for epoch in range(EPOCHS):
    print(f'\nTraining, epoch {epoch}')
    print('Learning Rate: ' + str(learning_rate))

    step += 1

    image_augmented, mask_augmented = augment_images()

    csv_logger = CSVLogger(progress_path, append=True, separator=';')

    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=BATCH_SIZE, validation_data=(x_valid, y_valid),
              verbose=1, callbacks=[csv_logger, tensorboard_callback])

    prediction_valid = model.predict(x_valid, batch_size=BATCH_SIZE, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid).numpy()
    print("Loss Validation: " + str(loss_valid))

    prediction_test = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    loss_test = dice_metric_loss(y_test, prediction_test).numpy()
    print("Loss Test: " + str(loss_test))

    with open(progressfull_path, 'a') as f:
        f.write(f'epoch: {epoch}\nval_loss: {loss_valid}\ntest_loss: {loss_test}\n\n\n')

    # Save only if loss_valid is lower than the current minimum
    if loss_valid < min_loss_for_saving:
        min_loss_for_saving = loss_valid
        print(f"Saved model with val_loss: {loss_valid}")
        model.save(model_save_path_h5)  # Save as .h5
        model.save(model_save_path_tf, save_format='tf')  # Save as TF SavedModel format

    del image_augmented
    del mask_augmented
    gc.collect()

# Computing the metrics and saving the results using the last saved model
print("Loading the fine-tuned model")
if not os.path.exists(model_save_path_h5):
    raise FileNotFoundError(f"Model file not found at {model_save_path_h5}. Check if any model was saved during training.")
model = tf.keras.models.load_model(model_save_path_h5, custom_objects={'dice_metric_loss': dice_metric_loss})

prediction_train = model.predict(x_train, batch_size=BATCH_SIZE, verbose=0)
prediction_valid = model.predict(x_valid, batch_size=BATCH_SIZE, verbose=0)
prediction_test = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
print("Predictions done")

dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))
print("Dice finished")

miou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
miou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
miou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))
print("Miou finished")

precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))
print("Precision finished")

recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))
print("Recall finished")

accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)), np.ndarray.flatten(prediction_train > 0.5))
accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)), np.ndarray.flatten(prediction_test > 0.5))
accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)), np.ndarray.flatten(prediction_valid > 0.5))
print("Accuracy finished")

final_file = 'results_' + model_type + '_' + str(filters) + '_' + dataset_type + '.txt'
print(final_file)

with open(final_file, 'a') as f:
    f.write(dataset_type + '\n\n')
    f.write(f'dice_train: {dice_train} dice_valid: {dice_valid} dice_test: {dice_test}\n\n')
    f.write(f'miou_train: {miou_train} miou_valid: {miou_valid} miou_test: {miou_test}\n\n')
    f.write(f'precision_train: {precision_train} precision_valid: {precision_valid} precision_test: {precision_test}\n\n')
    f.write(f'recall_train: {recall_train} recall_valid: {recall_valid} recall_test: {recall_test}\n\n')
    f.write(f'accuracy_train: {accuracy_train} accuracy_valid: {accuracy_valid} accuracy_test: {accuracy_test}\n\n\n\n')

print('File done')
print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
print("To visualize, run: tensorboard --logdir logs/")
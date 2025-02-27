import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, TensorBoard, Callback
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import DUCK_Net
from ImageLoader import ImageLoader2D
import random

# Checking the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Setting the model parameters
img_size = 352
dataset_type = 'kvasir'
learning_rate = 1e-4
seed_value = 42
filters = 17
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

ct = datetime.now()
model_type = "DuckNet"

# Paths
progress_path = f'ProgressFull/{dataset_type}_progress_csv_{model_type}_filters_{filters}_{ct}.csv'
os.makedirs(os.path.dirname(progress_path), exist_ok=True)
progressfull_path = f'ProgressFull/{dataset_type}_progress_{model_type}_filters_{filters}_{ct}.txt'
plot_path = f'ProgressFull/{dataset_type}_progress_plot_{model_type}_filters_{filters}_{ct}.png'
model_save_path_h5 = f'ModelSaveTensorFlow/kvasir/DuckNet_filters_17_finetuned_{ct}.h5'
model_save_path_folder = f'ModelSaveTensorFlow/kvasir/DuckNet_filters_17_finetuned_{ct}_folder'
os.makedirs(os.path.dirname(model_save_path_h5), exist_ok=True)
os.makedirs(model_save_path_folder, exist_ok=True)
pretrained_model_path = 'ModelSaveTensorFlow/DuckNet_17_Kvasir_Tf_Model'
tensorboard_log_dir = f'logs/finetune_{dataset_type}_{model_type}_filters_{filters}_{ct}'
os.makedirs(tensorboard_log_dir, exist_ok=True)

print(f"Model will be saved as .h5 to: {model_save_path_h5}")
print(f"Model will be saved as folder to: {model_save_path_folder}\n")

EPOCHS = 50
min_loss_for_saving = 0.2

# Loading the data
X, Y = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed_value)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle=True,
                                                      random_state=seed_value)

# Defining the augmentations
aug_train = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22),
                always_apply=True),
])


def augment_images():
    x_train_out, y_train_out = [], []
    for i in range(len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])
    return np.array(x_train_out), np.array(y_train_out)


# Custom callback for logging images and metrics
class CustomTensorBoardCallback(Callback):
    def __init__(self, log_dir, x_train, y_train, x_valid, y_valid, x_test, y_test, num_images=3):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.y_valid = x_valid, y_valid
        self.x_test, self.y_test = x_test, y_test
        # Adjust num_images to the smallest dataset size
        min_size = min(len(x_train), len(x_valid), len(x_test))
        self.num_images = min(num_images, max(1, min_size))  # Ensure at least 1, but no more than min_size
        print(f"Adjusted num_images to {self.num_images} based on smallest dataset size: {min_size}")
        self.image_indices = {
            'train': random.sample(range(len(x_train)), self.num_images) if len(x_train) > 0 else [],
            'valid': random.sample(range(len(x_valid)), self.num_images) if len(x_valid) > 0 else [],
            'test': random.sample(range(len(x_test)), self.num_images) if len(x_test) > 0 else []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            # Log scalar metrics
            tf.summary.scalar('train/loss', logs.get('loss'), step=epoch)
            tf.summary.scalar('valid/loss', logs.get('val_loss'), step=epoch)
            tf.summary.scalar('train/dice', logs.get('dice', 0), step=epoch)
            tf.summary.scalar('valid/dice', logs.get('val_dice', 0), step=epoch)
            tf.summary.scalar('train/precision', logs.get('precision'), step=epoch)
            tf.summary.scalar('valid/precision', logs.get('val_precision'), step=epoch)
            tf.summary.scalar('train/recall', logs.get('recall'), step=epoch)
            tf.summary.scalar('valid/recall', logs.get('val_recall'), step=epoch)
            tf.summary.scalar('train/accuracy', logs.get('accuracy'), step=epoch)
            tf.summary.scalar('valid/accuracy', logs.get('val_accuracy'), step=epoch)

            # Log images only if there are indices available
            for split in ['train', 'valid', 'test']:
                if not self.image_indices[split]:  # Skip if no indices (empty dataset)
                    continue
                x_data = self.x_train if split == 'train' else self.x_valid if split == 'valid' else self.x_test
                y_data = self.y_train if split == 'train' else self.y_valid if split == 'valid' else self.y_test
                pred = self.model.predict(x_data[self.image_indices[split]], verbose=0)
                # Ensure all arrays are 4D with 3 channels
                x_data_selected = x_data[self.image_indices[split]]  # Shape: [batch, height, width, 3]
                y_data_selected = y_data[
                    self.image_indices[split]]  # Shape: [batch, height, width] or [batch, height, width, 1]
                pred_selected = pred  # Shape: [batch, height, width] or [batch, height, width, 1]

                # Adjust dimensions if necessary
                if len(y_data_selected.shape) == 3:  # [batch, height, width]
                    y_data_selected = np.expand_dims(y_data_selected, axis=-1)  # [batch, height, width, 1]
                if len(pred_selected.shape) == 3:  # [batch, height, width]
                    pred_selected = np.expand_dims(pred_selected, axis=-1)  # [batch, height, width, 1]

                # Expand to 3 channels
                y_data_selected = np.repeat(y_data_selected, 3, axis=-1)  # [batch, height, width, 3]
                pred_selected = np.repeat(pred_selected, 3, axis=-1)  # [batch, height, width, 3]

                images = np.concatenate([x_data_selected, y_data_selected, pred_selected], axis=2)
                tf.summary.image(f'{split}/images', images, step=epoch, max_outputs=self.num_images)


# Loading and compiling the pre-trained model
print(f"Loading pre-trained model from: {pretrained_model_path}")
model = tf.keras.models.load_model(pretrained_model_path, custom_objects={'dice_metric_loss': dice_metric_loss})
model.compile(optimizer=optimizer, loss=dice_metric_loss,
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.MeanIoU(num_classes=2, name='miou')])

model.summary()

# Training the model
step = 0
for epoch in range(EPOCHS):
    print(f'Training, epoch {epoch}')
    print(f'Learning Rate: {learning_rate}')

    step += 1
    image_augmented, mask_augmented = augment_images()

    csv_logger = CSVLogger(progress_path, append=True, separator=';')
    tensorboard_callback = CustomTensorBoardCallback(tensorboard_log_dir, x_train, y_train, x_valid, y_valid, x_test,
                                                     y_test)

    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=4, validation_data=(x_valid, y_valid),
              verbose=1, callbacks=[csv_logger, tensorboard_callback])

    prediction_valid = model.predict(x_valid, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid).numpy()
    print(f"Loss Validation: {loss_valid}")

    prediction_test = model.predict(x_test, verbose=0)
    loss_test = dice_metric_loss(y_test, prediction_test).numpy()
    print(f"Loss Test: {loss_test}")

    with open(progressfull_path, 'a') as f:
        f.write(f'epoch: {epoch}\nval_loss: {loss_valid}\ntest_loss: {loss_test}\n\n\n')

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print(f"Saved model with val_loss: {loss_valid}")
        model.save(model_save_path_h5)  # Save as .h5
        model.save(model_save_path_folder, save_format='tf')  # Save as folder with .pb files

    del image_augmented, mask_augmented
    gc.collect()

# Computing and saving metrics
print("Loading the fine-tuned model")
model = tf.keras.models.load_model(model_save_path_h5, custom_objects={'dice_metric_loss': dice_metric_loss})

prediction_train = model.predict(x_train, batch_size=4)
prediction_valid = model.predict(x_valid, batch_size=4)
prediction_test = model.predict(x_test, batch_size=4)

# Calculate metrics
for split, pred, true in [('train', prediction_train, y_train),
                          ('valid', prediction_valid, y_valid),
                          ('test', prediction_test, y_test)]:
    dice = f1_score(np.ndarray.flatten(np.array(true, dtype=bool)), np.ndarray.flatten(pred > 0.5))
    miou = jaccard_score(np.ndarray.flatten(np.array(true, dtype=bool)), np.ndarray.flatten(pred > 0.5))
    precision = precision_score(np.ndarray.flatten(np.array(true, dtype=bool)), np.ndarray.flatten(pred > 0.5))
    recall = recall_score(np.ndarray.flatten(np.array(true, dtype=bool)), np.ndarray.flatten(pred > 0.5))
    accuracy = accuracy_score(np.ndarray.flatten(np.array(true, dtype=bool)), np.ndarray.flatten(pred > 0.5))

    print(f"{split.capitalize()} Metrics:")
    print(f"Dice: {dice}, Miou: {miou}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")

    with open(f'results_{model_type}_{filters}_{dataset_type}_finetuned.txt', 'a') as f:
        f.write(f'{split}:\n')
        f.write(f'dice: {dice}\nmiou: {miou}\nprecision: {precision}\nrecall: {recall}\naccuracy: {accuracy}\n\n')

print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
print("To visualize, run: tensorboard --logdir logs/")

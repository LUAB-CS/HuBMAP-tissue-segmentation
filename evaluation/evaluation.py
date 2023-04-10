"""
Here we will build a handful of function to visualize different metrics as well as the data itself
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ORGANS = ['kidney', 'largeintestine', 'lung', 'prostate', 'spleen']

def iou(y_true, y_pred):
    intersection = np.count_nonzero(y_true * y_pred)
    union = np.count_nonzero(y_true + y_pred)
    return intersection / union

# Predictions and true labels of validation dataset
def get_y_true_y_pred(val_pred):
    thresholds = np.arange(0, 1.01, 0.01)
    IoUs = {}
    for t in thresholds:
        IoUs[t] = []

    IoUsOrgans = {}
    for o in ORGANS:
        IoUsOrgans[o] = {}
        for t in thresholds:
            IoUsOrgans[o][t] = []
    print(tqdm)
    for idx, image in enumerate(tqdm(val_pred['val_images'])):
        y_true = val_pred['val_masks'][idx]
        organ = val_pred['val_organs'][idx]
        y_pred = val_pred['val_y_preds'][idx]

        if idx == 0:
            print(f'image shape: {image.shape}, y_true shape: {y_true.shape}')
            print(f'organs: {organ}, y_pred shape: {y_pred.shape}')

        # Compute IoU for each threshold
        for t in thresholds:
            IoU = iou(y_true, (y_pred > t).astype(np.int8))
            IoUs[t].append(IoU)
            IoUsOrgans[organ][t].append(IoU)

    return IoUs, IoUsOrgans

def plot_iou_by_threshold(ious, name):
    thresholds = list(ious.keys())
    MeanIoUs = [np.mean(v)for v in ious.values()]

    plt.figure(figsize=(12,8))
    plt.title(f'Mean IoU by Threshold {name}', size=24)
    plt.plot(thresholds, MeanIoUs)
    plt.grid()
    plt.xlabel('Threshold', size=16)
    plt.ylabel('Mean IoU', size=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.ylim(0,1)

    # Best Threshold
    arg_best = np.argmax(MeanIoUs)
    threshold_best = thresholds[arg_best]
    mean_iou_best = MeanIoUs[arg_best]
    plt.scatter(threshold_best, mean_iou_best, color='red', s=100, marker='o', label=f'Best Mean IoU ({mean_iou_best:.3f}) at Threshold {threshold_best:.3f}')
    plt.legend(prop={'size': 16})

    plt.show()

    # Save Best Threshold
    np.save(f'threshold_best_{name}.npy', threshold_best)

    return threshold_best


def plot_validation_predictions(v, threshold_best, n, IoUs):
    """
    Code inspo from : https://www.kaggle.com/code/markwijkhuizen/hubmap-hpa-hacking-the-human-body-training
    TO UPDATE WHEN MODEL IS OUT
    Input :
    v : validation dataset contatine : images, GT masks, organ, predicted mask
    threshold_best : threshold for predicted mask (each pixel is assigned a probability to be part of the mask)
    n : how many examples to show
    """
    for idx in range(n):
        image = v['val_images'][idx]
        y_true = v['val_masks'][idx]
        organ = v['val_organs'][idx]
        y_pred = v['val_y_preds'][idx]

        organ = organ
        # Predicted Mask
        y_pred_binary = (y_pred > threshold_best).astype(np.uint8)
        # Red = False Positive
        r = ((y_pred_binary == 1) * (y_true == 0)).astype(np.uint8)
        # Green = True Positive
        g = ((y_pred_binary == 1) * (y_true == 1)).astype(np.uint8)
        # Blue = False Negative
        b = ((y_pred_binary == 0) * (y_true == 1)).astype(np.uint8)
        # Error Visualization Using RGB
        mask_error = np.stack((r, g, b), axis=0).squeeze() * 255

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image.transpose((1,2,0)))
        axes[0].set_title(f'Image {organ} IoU: {IoUs[threshold_best][idx]:.2f}')
        axes[1].imshow(y_true.transpose((1,2,0)))
        axes[1].set_title('Mask True')
        axes[2].imshow(y_pred.transpose((1,2,0)))
        axes[2].set_title('Mask Pred')
        axes[3].imshow(y_pred_binary.transpose((1,2,0)))
        axes[3].set_title('Mask Pred Binary')
        axes[4].imshow(mask_error.transpose((1,2,0)))
        axes[4].set_title('Mask Error')

        plt.show()
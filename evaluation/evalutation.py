"""
Here we will build a handful of function to visualize different metrics as well as the data itself
"""
import numpy as np
import matplotlib.pyplot as plt

IoUs = [] # TO BUILD, compute IoUs for different value of the threshold
def plot_validation_predictions(v, threshold_best, n):
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

        organ = organ.decode()
        # Predicted Mask
        y_pred_binary = (y_pred > threshold_best).astype(np.uint8)
        # Red = False Positive
        r = ((y_pred_binary == 1) * (y_true == 0)).astype(np.uint8)
        # Green = True Positive
        g = ((y_pred_binary == 1) * (y_true == 1)).astype(np.uint8)
        # Blue = False Negative
        b = ((y_pred_binary == 0) * (y_true == 1)).astype(np.uint8)
        # Error Visualization Using RGB
        mask_error = np.stack((r, g, b), axis=2).squeeze() * 255

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image)
        axes[0].set_title(f'Image {organ} IoU: {IoUs[threshold_best][idx]:.2f}')
        axes[1].imshow(y_true)
        axes[1].set_title('Mask True')
        axes[2].imshow(y_pred)
        axes[2].set_title('Mask Pred')
        axes[3].imshow(y_pred_binary)
        axes[3].set_title('Mask Pred Binary')
        axes[4].imshow(mask_error)
        axes[4].set_title('Mask Error')

        plt.show()
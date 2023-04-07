"""
Here, I make for assumption for now that the model outputs are binary segmentation arrays of the size of the
original picture.
"""

import numpy as np
import pandas as pd


def make_submission(model, test_df, test_dir, threshold):
    for i,row in test_df.iterrows():
        try:
            idx = row["id"]
            img_path = img_path = f"{test_dir}/{idx}.tiff"
            # Predicted Mask
            y_pred_binary = (y_pred > threshold_best).astype(np.uint8)
            mask = model(img_path=img_path,threshold = 220,plot=True)
    #         gt_mask = rleToMask(row['rle'],row["img_width"],row["img_height"])
    #         dice_coeff.append(dice(mask,gt_mask))

            # if i>4:
            #     break
            plt.imshow(mask,cmap = "gray")
            rles.append(rle_encode(mask))

        except:
            print(f"error in {i}")
            continue
    prediction = model(test_dataset) # I will update the beginning of the script to fit the other's work
    submission = {'id':[], 'rle':[]}

    # From binary arrays to run-length encoding
    linestart = None
    for id, mask in enumerate(prediction): # Depend on the test dataloader
        mask_size = mask.shape[0]
        submission["id"].append(id)
        rle = []
        binary_mask = (mask> threshold).astype(np.uint8)
        for k, line in enumerate(binary_mask):
            for l, pixel in enumerate(line):
                if pixel == 0 and linestart : # End current segment
                    lineend = k * mask_size + l - 1
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                if linestart and l == mask_size - 1:# Segment end at border
                    lineend = k * mask_size + l
                    rle = rle + f"{linestart} {lineend} "
                    linestart = None
                elif pixel == 1 and not(linestart) : # Start new segment
                    linestart = k * mask_size

    # Create .csv
    test_df = pd.read_csv("HuBMAP-tissue-segmentation/data/" + "test.csv")
    sub = pd.DataFrame(submission)
    test_df = test_df.merge(sub, on='id')
    sub = test_df[['id', 'rle']].copy()
    sub.to_csv('submission.csv', index=False)
    # sub

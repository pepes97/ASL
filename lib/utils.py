
import matplotlib.pyplot as plt
import numpy as np

def show_batch(image_batch, label_batch, classnames):

    """
        Args:
        - image_batch: batch of images (not labels)
        - label_batch: batch of labels 
    """
    plt.figure(figsize=(12,12))
    for n in range(64):
        ax = plt.subplot(8,8,n+1)
        plt.imshow(image_batch[n])
        idx = np.where(label_batch[n] == 1)[0][0]
        plt.title(classnames[idx])
        plt.axis('off')

def show_predicted_batch(image_batch, label_batch, asl_model, classnames):
  plt.figure(figsize=(12,12))
  for n in range(36):
      ax = plt.subplot(6,6,n+1)
      plt.imshow(image_batch[n])
      pred = asl_model(image_batch)
      idx = np.where(label_batch[n] == 1)[0][0]
      idx_pred = pred[n].numpy().argmax()
      plt.title(f"{classnames[idx]} -> {classnames[idx_pred]}")
      plt.axis('off')

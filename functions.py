import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def plot_digit(data):
    '''
    This function resizes and plots one digit
    '''
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

    

def plot_digits(instances, images_per_row=10, **options):
    '''
    This function plots 10 digits by 10 digits
    '''
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    '''
    This function computes the precision and recall scores for all possible thresholds
    '''
    #changing figure size
    plt.figure(figsize=(8, 4))
    #plot the precision line
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    #plot the recall line
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    #add a legend
    plt.legend(loc="center right", fontsize=16) 
    #label the x-axis
    plt.xlabel("Threshold", fontsize=16)
    #configure the gridline
    plt.grid(True)         
    #set the axis labels
    plt.axis([-50000, 50000, 0, 1])             


    #finding the 90% precision spot
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    #finding the 90% threshold spot
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

    #plot vertical line for 90% precision 
    plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")  
    #plot horizontal line for 90% precision
    plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")   
    #plot horizontal line for 90% threshold
    plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
    #plot point of 90% precision
    plt.plot([threshold_90_precision], [0.9], "ro")    
    #plot point of 90% threshold
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")                                                                        # Not shown
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    '''
    This function plots the ROC curve
    '''
    #changing figure size
    plt.figure(figsize=(8, 6))
    #plotting precision vs recall line
    plt.plot(recalls, precisions, "b-", linewidth=2)
    #labeling and configuring axis
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    
    #finding the 90% precision spot
    recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
    #plotting the 90% precision spot
    plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
    plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
    plt.plot([recall_90_precision], [0.9], "ro")
    plt.show()



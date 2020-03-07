
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix_compute(predicts, labels, num_classes):
    
    predicts_flat = predicts.flatten()
    labels_flat = labels.flatten()

    cm = np.zeros((num_classes,num_classes))

    accuracy = 0

    for i in range(predicts_flat.shape[0]):
        predict = int(predicts_flat[i])
        label = int(labels_flat[i])
        cm[label:label+1, predict:predict+1] += 1
        
        if label == predict:
            accuracy += 1

    accuracy = round(accuracy / len(labels_flat),4)

    return cm, accuracy

def confusion_matrix_save(cm, path='', name='Confusion.png'):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="w")

    plt.savefig(path + name)

# cm = np.array([[0,1], [2,3]])
# l = np.array([[0,1], [2,3]])
# r = confusion_matrix_compute(cm,l)
# confusion_matrix_save(r)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def show_cm():
    with open("confusion_mat/1_400X_V_confusion_matrix.pkl", "rb") as f:
        confusion_matrix_data = pickle.load(f)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
show_cm()

    



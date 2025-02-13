import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Предполагая наличие метрик обучения:
train_accuracies = []
test_accuracies = []

for i in range(num_iterations):
    # ... логика обучения модели ...
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracy Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

def plot_training_results(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()



def plot_final_results(test_results):
    plt.figure(figsize=(10, 5))
    plt.plot(test_results, label='Test Loss')
    plt.xlabel('Sample')
    plt.ylabel('Loss')
    plt.title('Test Loss on Final Model')
    plt.show()
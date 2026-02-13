import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Paths
base_dir = os.path.dirname(__file__)
model_paths = {
    'ann': os.path.join(base_dir, 'cifar100_ann_best.h5'),
    'basic_cnn': os.path.join(base_dir, 'cifar100_basic_cnn_best.h5'),
    'deeper_cnn': os.path.join(base_dir, 'cifar100_deeper_cnn_best.h5'),
}

# Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
# normalize
x_test = x_test.astype('float32') / 255.0

results = {}

for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"Model file not found: {path}, skipping {name}")
        continue
    print(f"Loading model {name} from {path}")
    model = load_model(path)
    loss, acc = model.evaluate(x_test, np.eye(100)[y_test.flatten()], verbose=0)
    print(f"{name}: loss={loss:.4f}, acc={acc:.4f}")

    # Predictions
    preds = model.predict(x_test, batch_size=128, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test.flatten()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    results[name] = {
        'loss': float(loss),
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix_shape': [len(cm), len(cm[0]) if len(cm)>0 else 0]
    }

# Save summary JSON
out_json = os.path.join(base_dir, 'cifar100_evaluation_summary.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print('Saved evaluation summary to', out_json)

# Create a simple accuracy bar plot
names = []
accs = []
for name in results:
    names.append(name)
    accs.append(results[name]['accuracy'])

if names:
    plt.figure(figsize=(6,4))
    plt.bar(names, accs, color=['#1f77b4','#ff7f0e','#2ca02c'])
    plt.ylim(0,1)
    plt.ylabel('Test Accuracy')
    plt.title('CIFAR-100 Model Test Accuracy')
    plot_path = os.path.join(base_dir, 'cifar100_accuracy.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print('Saved accuracy plot to', plot_path)

# Save a small grid of sample predictions from deeper model if available
if 'deeper_cnn' in results and os.path.exists(model_paths['deeper_cnn']):
    model = load_model(model_paths['deeper_cnn'])
    sample_idx = np.random.choice(len(x_test), size=25, replace=False)
    imgs = x_test[sample_idx]
    preds = model.predict(imgs)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test.flatten()[sample_idx]

    plt.figure(figsize=(10,10))
    for i, idx in enumerate(range(25)):
        plt.subplot(5,5,i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(f"T:{y_true[i]}\nP:{y_pred[i]}", fontsize=8)
    sample_path = os.path.join(base_dir, 'cifar100_deeper_predictions_sample.png')
    plt.savefig(sample_path, bbox_inches='tight')
    plt.close()
    print('Saved sample prediction grid to', sample_path)
else:
    print('Deeper model not available for sample predictions')

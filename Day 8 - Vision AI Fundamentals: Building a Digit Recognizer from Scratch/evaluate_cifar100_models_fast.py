import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Fast evaluation: limit test samples to speed up on CPU
MAX_SAMPLES = 5000

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

# Subsample for speed if needed
if MAX_SAMPLES and MAX_SAMPLES < len(x_test):
    idx = np.random.choice(len(x_test), size=MAX_SAMPLES, replace=False)
    x_test_small = x_test[idx]
    y_test_small = y_test[idx]
else:
    x_test_small = x_test
    y_test_small = y_test

results = {}

for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"Model file not found: {path}, skipping {name}")
        continue
    print(f"Loading model {name} from {path}")
    model = load_model(path)
    try:
        loss, acc = model.evaluate(x_test_small, np.eye(100)[y_test_small.flatten()], verbose=0, batch_size=256)
    except Exception as e:
        print('Evaluation failed for', name, 'error:', e)
        # fallback to predict+accuracy
        preds = model.predict(x_test_small, batch_size=256, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = y_test_small.flatten()
        acc = accuracy_score(y_true, y_pred)
        loss = None

    print(f"{name}: loss={loss}, acc={acc}")

    # Predictions for report
    preds = model.predict(x_test_small, batch_size=256, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test_small.flatten()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results[name] = {
        'loss': float(loss) if loss is not None else None,
        'accuracy': float(acc),
        'classification_report_summary': { 'macro avg': report.get('macro avg', {}), 'weighted avg': report.get('weighted avg', {}) },
    }

# Save summary JSON
out_json = os.path.join(base_dir, 'cifar100_evaluation_summary_fast.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print('Saved fast evaluation summary to', out_json)

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
    plt.title('CIFAR-100 Model Test Accuracy (sampled)')
    plot_path = os.path.join(base_dir, 'cifar100_accuracy_fast.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print('Saved accuracy plot to', plot_path)

# Save a small grid of sample predictions from deeper model if available
if 'deeper_cnn' in results and os.path.exists(model_paths['deeper_cnn']):
    model = load_model(model_paths['deeper_cnn'])
    sample_idx = np.random.choice(len(x_test_small), size=min(25, len(x_test_small)), replace=False)
    imgs = x_test_small[sample_idx]
    preds = model.predict(imgs, batch_size=64)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test_small.flatten()[sample_idx]

    plt.figure(figsize=(10,10))
    for i in range(len(imgs)):
        plt.subplot(5,5,i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(f"T:{y_true[i]}\nP:{y_pred[i]}", fontsize=8)
    sample_path = os.path.join(base_dir, 'cifar100_deeper_predictions_sample_fast.png')
    plt.savefig(sample_path, bbox_inches='tight')
    plt.close()
    print('Saved sample prediction grid to', sample_path)
else:
    print('Deeper model not available for sample predictions')

import argparse
import numpy as np
import os
import pandas as pd
os.environ['KERAS_BACKEND'] = 'torch'
import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle


parser = argparse.ArgumentParser(description='Train neural network for crystal structure prediction')
parser.add_argument('--prediction', type=str, default='lattice_system',
                    choices=['lattice_system', 'bravais_lattice', 'reindexed_spacegroup_symbol_hm'],
                    help='Property to predict (default: lattice_system)')
parser.add_argument('--tag', type=str, default='sa',
                    help='Model tag/version identifier (default: sa)')

args = parser.parse_args()

prediction = args.prediction
tag = args.tag

# Model configuration
HIDDEN_LAYERS = [1024, 512, 256, 128, 64, 32]  # List of hidden layer sizes
ACTIVATION = 'gelu'
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 40

base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex'
output_dir = '/global/cfs/cdirs/m4064/dwmoreau/publication_materials/classification'
model_dir = os.path.join(base_dir, 'models')
n_max = 100000
n_peaks = 20


def load_data(model_dir, name, prediction, n_peaks, n_max=10000, seed=123):
    print(f'Loading {name}')
    columns = ['lattice_system', 'train', 'augmented', 'q2', 'bravais_lattice']
    data_file_name = os.path.join(model_dir, name, 'data', 'data.parquet')
    df = pd.read_parquet(data_file_name, columns=columns)
    points = df['q2']
    indices = points.apply(len) >= n_peaks
    df = df.loc[indices]

    unaugmented = df[df['augmented'] == False]
    augmented = df[df['augmented'] == True]

    bravais_lattices = df['bravais_lattice'].unique()
    if prediction == 'bravais_lattice':
        n_per_bl = n_max
    else:
        n_per_bl = int(n_max / len(bravais_lattices))
    sampled_df = []  # Should be a list to collect DataFrames
    for bl in bravais_lattices:
        unaugmented_bl = unaugmented[unaugmented['bravais_lattice'] == bl]
        augmented_bl = augmented[augmented['bravais_lattice'] == bl]
        
        if len(unaugmented_bl) >= n_per_bl:
            sampled_df.append(unaugmented_bl.sample(n=n_per_bl, random_state=seed))
        else:
            # Take all unaugmented and fill the rest from augmented
            remaining = n_per_bl - len(unaugmented_bl)
            sampled_augmented = augmented_bl.sample(n=min(remaining, len(augmented_bl)), random_state=seed)
            sampled_df.append(pd.concat([unaugmented_bl, sampled_augmented]))
    sampled_df = pd.concat(sampled_df, ignore_index=True)  # Add ignore_index=True

    # Shuffle the final result
    sampled_df = sampled_df.drop('augmented', axis=1)  # Drop bravais_lattice too
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled_df


def build_model(input_dim, num_classes):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for hidden_size in HIDDEN_LAYERS:
        model.add(layers.Dense(hidden_size, activation='gelu', use_bias=False))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_confusion_matrix(cm, classes, ax, title, normalize=True):
    """Plot confusion matrix on given axes"""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
    else:
        cm_display = cm
        fmt = 'd'
    
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='%' if normalize else 'Count')
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           xlabel='Predicted',
           ylabel='True',
           title=title)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")


# Combine all data
df = pd.concat([
    #load_data(model_dir, f'cubic_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'tetragonal_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'hexagonal_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'rhombohedral_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'orthorhombic_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'monoclinic_{tag}', prediction, n_peaks, n_max),
    load_data(model_dir, f'triclinic_{tag}', prediction, n_peaks, n_max),
], ignore_index=True)

# Or to see unique values with their counts:
print(df['bravais_lattice'].value_counts())

# Prepare features and labels
X = np.array([x[:n_peaks] for x in df['q2'].values])
y = df[prediction].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data using 'train' column
train_mask = df['train'].values
X_train = X[train_mask]
y_train = y_encoded[train_mask]
X_val = X[~train_mask]
y_val = y_encoded[~train_mask]

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build and train model
num_classes = len(label_encoder.classes_)
model = build_model(input_dim=n_peaks, num_classes=num_classes)

#print(model.summary())

history = model.fit(
    X_train_scaled, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_scaled, y_val),
    verbose=1
)

# Evaluate
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
print(f"\nValidation accuracy: {val_accuracy:.4f}")

# Predictions for both train and validation sets
y_train_pred = model.predict(X_train_scaled)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

y_val_pred = model.predict(X_val_scaled)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Compute confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred_classes)
cm_val = confusion_matrix(y_val, y_val_pred_classes)

print("\nTraining Confusion Matrix:")
print(cm_train)
print("\nValidation Confusion Matrix:")
print(cm_val)

# Classification reports
print("\nTraining Classification Report:")
print(classification_report(y_train, y_train_pred_classes, target_names=label_encoder.classes_))
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_classes, target_names=label_encoder.classes_))

# Save confusion matrix data and scaler
cm_data = {
    'cm_train': cm_train,
    'cm_val': cm_val,
    'classes': label_encoder.classes_,
    'y_train': y_train,
    'y_train_pred': y_train_pred_classes,
    'y_val': y_val,
    'y_val_pred': y_val_pred_classes,
    'scaler': scaler,
    'label_encoder': label_encoder
}

with open(os.path.join(output_dir, f'confusion_data_{prediction}_{tag}.pkl'), 'wb') as f:
    pickle.dump(cm_data, f)

# Plot confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_confusion_matrix(cm_train, label_encoder.classes_, axes[0], 'Training Set', normalize=True)
plot_confusion_matrix(cm_val, label_encoder.classes_, axes[1], 'Validation Set', normalize=True)

fig.tight_layout()
plt.savefig(
    os.path.join(output_dir, f'confusion_matrices_{prediction}_{tag}.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# Loss plot
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(
    os.path.join(output_dir, f'training_history_{prediction}_{tag}.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

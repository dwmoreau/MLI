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
    sampled_df = []
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
    sampled_df = pd.concat(sampled_df, ignore_index=True)

    sampled_df = sampled_df.drop('augmented', axis=1)
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled_df


def load_opxrd_data(n_peaks):
    df = pd.read_json(
        '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/opxrd/CNRS_output_data_verified_final3.json'
    )[['bravais_lattice', 'lattice_system', 'peak_positions']]
    df = df[df['lattice_system'] != 'cubic']
    q2 = []
    for i in range(len(df)):
        q2_entry = [q**2 for q in df.iloc[i]['peak_positions']]
        q2.append(q2_entry[:n_peaks])
    df['q2'] = q2
    return df
    

class data_manager:
    def __init__(self, df, n_peaks, scaler=None, label_encoder=None):
        self.df = df
        # Prepare features and labels
        self.X = np.array([x[:n_peaks] for x in self.df['q2'].values])
        self.y = self.df[prediction].values
        
        # Encode labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.y)
            self.num_classes = len(self.label_encoder.classes_)
        else:
            self.label_encoder = label_encoder
        self.y_encoded = self.label_encoder.transform(self.y)

        # Split data using 'train' column if available
        if 'train' in self.df.keys():
            self.train = True
            train_mask = self.df['train'].values
            self.X_train = self.X[train_mask]
            self.y_train = self.y_encoded[train_mask]
            self.X_val = self.X[~train_mask]
            self.y_val = self.y_encoded[~train_mask]
        else:
            self.train = False

        # Fix: fit scaler on appropriate data depending on whether train split exists
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X_train if self.train else self.X)
        else:
            self.scaler = scaler

        if self.train:
            self.X_train_scaled = self.scaler.transform(self.X_train)
            self.X_val_scaled = self.scaler.transform(self.X_val)
        else:
            self.X_scaled = self.scaler.transform(self.X)


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


def evaluate(X_scaled, y, model, label_encoder, scaler, tag, data_source):
    loss, accuracy = model.evaluate(X_scaled, y)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    y_pred = model.predict(X_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Fix: use y instead of undefined y_train
    cm = confusion_matrix(y, y_pred_classes)

    print(f"\n{data_source} Confusion Matrix:")
    print(cm)

    print(f"\n{data_source} Classification Report:")
    print(classification_report(y, y_pred_classes, target_names=label_encoder.classes_))

    # Fix: use cm instead of undefined cm_
    cm_data = {
        'cm': cm,
        'classes': label_encoder.classes_,
        'y': y,
        'y_pred': y_pred_classes,
        'scaler': scaler,
        'label_encoder': label_encoder
    }

    with open(os.path.join(output_dir, f'confusion_data_{prediction}_{tag}_{data_source}.pkl'), 'wb') as f:
        pickle.dump(cm_data, f)
    return cm


parser = argparse.ArgumentParser(description='Train neural network for crystal structure prediction')
parser.add_argument(
    '--prediction',
    type=str,
    default='lattice_system',
    choices=['lattice_system', 'bravais_lattice', 'reindexed_spacegroup_symbol_hm'],
    help='Property to predict (default: lattice_system)'
)
parser.add_argument(
    '--tag',
    type=str,
    default='sa',
    help='Model tag/version identifier (default: sa)'
)

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

df_opxrd = load_opxrd_data(n_peaks)

synthetic_manager = data_manager(df, n_peaks)
# Fix: pass df_opxrd instead of df
opxrd_manager = data_manager(
    df_opxrd,
    n_peaks,
    scaler=synthetic_manager.scaler,
    label_encoder=synthetic_manager.label_encoder
)

print(df['bravais_lattice'].value_counts())
print(df_opxrd['bravais_lattice'].value_counts())

model = build_model(input_dim=n_peaks, num_classes=synthetic_manager.num_classes)

history = model.fit(
    synthetic_manager.X_train_scaled,
    synthetic_manager.y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(
        synthetic_manager.X_val_scaled,
        synthetic_manager.y_val
    ),
    verbose=1
)

cm_train = evaluate(
    X_scaled=synthetic_manager.X_train_scaled,
    y=synthetic_manager.y_train,
    model=model,
    label_encoder=synthetic_manager.label_encoder,
    scaler=synthetic_manager.scaler,
    tag=tag,
    data_source='Train',
)
# Fix: data_source corrected from 'Train' to 'Val'
cm_val = evaluate(
    X_scaled=synthetic_manager.X_val_scaled,
    y=synthetic_manager.y_val,
    model=model,
    label_encoder=synthetic_manager.label_encoder,
    scaler=synthetic_manager.scaler,
    tag=tag,
    data_source='Val',
)
# Fix: data_source corrected from 'Train' to 'opXRD'
cm_opxrd = evaluate(
    X_scaled=opxrd_manager.X_scaled,
    y=opxrd_manager.y_encoded,
    model=model,
    label_encoder=opxrd_manager.label_encoder,
    scaler=opxrd_manager.scaler,
    tag=tag,
    data_source='opXRD',
)

# Fix: use synthetic_manager.label_encoder and opxrd_manager.label_encoder
# instead of undefined label_encoder
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_confusion_matrix(cm_train, synthetic_manager.label_encoder.classes_, axes[0], 'Training Set', normalize=True)
plot_confusion_matrix(cm_val, synthetic_manager.label_encoder.classes_, axes[1], 'Validation Set', normalize=True)
plot_confusion_matrix(cm_opxrd, opxrd_manager.label_encoder.classes_, axes[2], 'opXRD', normalize=True)
fig.tight_layout()
plt.savefig(
    os.path.join(output_dir, f'confusion_matrices_{prediction}_{tag}.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
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
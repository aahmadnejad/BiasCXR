from dataclasses import dataclass
import os

@dataclass
class Configs:
    EMBEDDING_DIM = 1376
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 1
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256
    OUTPUT_DIR = "output"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "cxr_model.pt")
    DEBIASED_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "cxr_model_debiased.pt")

    LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    AGE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 150)]
    AGE_BIN_LABELS = ['0-20', '20-40', '40-60', '60-80', '80+']
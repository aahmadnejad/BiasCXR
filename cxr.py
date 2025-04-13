from torch.utils.data import Dataset
import numpy as np
import torch

from reader import read_tfrecord
from processor import safe_convert_labels
from store import Configs

configs = Configs()

class CXRDataset(Dataset):
    def __init__(self, dataframe, embedding_transform=None):
        self.df = dataframe
        self.embedding_transform = embedding_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            try:
                path = row['path']
                if path.endswith('.npy'):
                    embedding = np.load(path)
                elif path.endswith('.tfrecord'):
                    embedding = read_tfrecord(path)
                else:
                    print(f"Unsupported file format: {path}")
                    embedding = np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)
            except Exception as e:
                print(f"Error loading embedding at index {idx}: {e}")
                embedding = np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)
            if self.embedding_transform:
                embedding = self.embedding_transform(embedding)
            features = torch.tensor(embedding, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
            labels = safe_convert_labels(row, configs.LABELS)
            additional_features = []
            if 'gender' in self.df.columns:
                try:
                    gender_feature = 1.0 if row['gender'] == 'M' else 0.0
                    additional_features.append(gender_feature)
                except:
                    additional_features.append(0.0)
            if 'age' in self.df.columns:
                try:
                    age_feature = float(row['age']) / 100.0
                    additional_features.append(age_feature)
                except:
                    additional_features.append(0.0)
            if 'anchor_age' in self.df.columns:
                try:
                    for group in configs.AGE_BIN_LABELS:
                        if row['anchor_age'] == group:
                            additional_features.append(1.0)
                        else:
                            additional_features.append(0.0)
                except:
                    additional_features.extend([0.0] * len(configs.AGE_BIN_LABELS))
            if 'race' in self.df.columns:
                try:
                    race_categories = self.df['race'].unique()
                    for race in race_categories:
                        if row['race'] == race:
                            additional_features.append(1.0)
                        else:
                            additional_features.append(0.0)
                except:
                    if 'race' in self.df.columns:
                        additional_features.extend([0.0] * len(self.df['race'].unique()))
            if additional_features:
                additional_features = torch.tensor(additional_features, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
                features = torch.cat([features, additional_features])

            return features, labels

        except Exception as e:
            print(f"Critical error in __getitem__ at index {idx}: {e}")
            dummy_features = torch.zeros(configs.EMBEDDING_DIM, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
            dummy_labels = torch.zeros(len(configs.LABELS), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
            return dummy_features, dummy_labels
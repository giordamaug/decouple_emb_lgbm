import torch
import numpy as np
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel


# =========================================================
# 1. CLINICALBERT ENCODER (WITH CACHE)
# =========================================================

class ClinicalBERTEncoder:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.cache = {}

    @torch.no_grad()
    def encode(self, text):
        if text in self.cache:
            return self.cache[text]

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        output = self.model(**inputs)
        emb = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        self.cache[text] = emb
        return emb


# =========================================================
# 2. DATASET
# =========================================================

class ClinicalTextDataset:
    def __init__(self, sequences_dict, labels_dict=None):
        self.ids = list(sequences_dict.keys())
        self.sequences = [sequences_dict[pid] for pid in self.ids]
        self.labels = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = self.sequences[idx]

        if self.labels is None:
            return pid, x
        return pid, x, self.labels[pid]


# =========================================================
# 3. SEQUENCE EMBEDDING BUILDER
# =========================================================

def build_patient_embeddings(sequences_dict, encoder):
    out = {}

    for pid, seq in sequences_dict.items():
        emb_seq = [encoder.encode(event) for event in seq]
        out[pid] = np.vstack(emb_seq)

    return out


# =========================================================
# 4. SEQUENCE ENCODERS
# =========================================================

class SimpleRETAIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.att = torch.nn.Linear(hidden_dim, 1)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        alpha = torch.softmax(self.att(h), dim=1)
        context = (alpha * h).sum(dim=1)
        logits = self.classifier(context)
        return logits, context


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


def mean_pool(x):
    return x.mean(dim=1)


# =========================================================
# 5. LIGHTGBM TRAINER
# =========================================================

def train_lgbm(X_train, y_train, X_valid, y_valid):
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc"
    )

    return model


# =========================================================
# 6. PATIENT EMBEDDING FROM SEQUENCES
# =========================================================

def encode_patients(sequence_embeddings, model=None, mode="mean"):
    X, ids = [], []

    for pid, seq in sequence_embeddings.items():
        x = torch.tensor(seq, dtype=torch.float)

        if mode == "mean":
            emb = x.mean(dim=0)

        elif mode == "retain":
            x = x.unsqueeze(0)
            with torch.no_grad():
                _, emb = model(x)
            emb = emb.squeeze(0)

        elif mode == "lstm":
            x = x.unsqueeze(0)
            with torch.no_grad():
                emb = model(x)
            emb = emb.squeeze(0)

        else:
            raise ValueError("Unknown mode")

        X.append(emb.cpu().numpy())
        ids.append(pid)

    return np.vstack(X), ids


# =========================================================
# 7. FULL PIPELINE
# =========================================================

def run_pipeline(train_seq, valid_seq, y_train, y_valid,
                 mode="mean"):

    encoder = ClinicalBERTEncoder()

    train_emb_seq = build_patient_embeddings(train_seq, encoder)
    valid_emb_seq = build_patient_embeddings(valid_seq, encoder)

    if mode == "retain":
        seq_model = SimpleRETAIN(input_dim=768, hidden_dim=128)
    elif mode == "lstm":
        seq_model = LSTMEncoder(input_dim=768, hidden_dim=128)
    else:
        seq_model = None

    X_train, _ = encode_patients(train_emb_seq, seq_model, mode=mode)
    X_valid, _ = encode_patients(valid_emb_seq, seq_model, mode=mode)

    clf = train_lgbm(X_train, y_train, X_valid, y_valid)

    return clf
import pandas as pd
import numpy as np
from .models import FlexibleLSTMModel, LSTMDataset, lstm_collate_fn
from .models import BEHRTDataset, BEHRTModel
from .models import RETAINModel, RETAINDataset, visit_collate_fn
from .models import GRUModel, GRUDDataset, GRUDModel, grud_collate_fn
from .models import DipoleDataset, DipoleModel, dipole_collate
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output

plotsize = (4,3)

def FlexLSTMclassifier(sequences,
                 labels, 
                 word_to_idx=None, 
                 train_idx=None, 
                 valid_idx=None,
                 num_epochs=10,
                 batch_size=32,
                 embed_size=64,
                 hidden_size=128,
                 enable_plot=False,
                 frame_tqdm=None,
                 frame_plot=None,
                 pooling=False,
                 use_padding=False,
                 bidirectional=False,
                 use_attention=False
                 ):

    y_train = {id: val for (id,val) in labels.items() if id in train_idx} 
    y_valid = {id: val for (id,val) in labels.items() if id in valid_idx}
    train_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in train_idx}
    train_dataset = LSTMDataset(train_sentences, labels_dict=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)
    test_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in valid_idx}
    test_dataset = LSTMDataset(test_sentences, labels_dict=y_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)

    model = FlexibleLSTMModel(len(word_to_idx), embed_dim=embed_size, 
                              hidden_dim=hidden_size, 
                              pooling=pooling,
                              use_padding=use_padding, 
                              bidirectional=bidirectional,
                              use_attention=use_attention)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader)
    return y_pred, y_prob

def RETAINclassifier(sequences, 
                    labels,
                    word_to_idx=None, 
                    train_idx=None, 
                    valid_idx=None,
                    num_epochs=10,
                    batch_size=32,
                    embed_size=64,
                    hidden_size=128,
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                 ):
    train_sequences = [[[word_to_idx[word] for word in words] for words,_ in sequences[id]] for id in train_idx]
    test_sequences = [[[word_to_idx[word] for word in words] for words,_ in sequences[id]] for id in valid_idx]
    y_train = np.array([val for (id,val) in labels.items() if id in train_idx], dtype=np.float32)
    y_valid = np.array([val for (id,val) in labels.items() if id in valid_idx], dtype=np.float32)
    
    #max_code = max(map(lambda p: max(map(lambda v: max(v), p)), train_sequences + test_sequences))
    num_features = len(word_to_idx) #max_code + 1
    # Inizializza e allena modello LSTM unidirezionale
    model = RETAINModel(len(word_to_idx), dim_emb=embed_size, dim_alpha=hidden_size, dim_beta=hidden_size)
    test_dataset = RETAINDataset(test_sequences, y_valid, num_features)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=visit_collate_fn)
    train_dataset = RETAINDataset(train_sequences, y_train, num_features)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=visit_collate_fn)
    model.train_model(train_dataloader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    
    train_patient_embeddings, tids = model.get_embeddings(train_dataloader)
    test_patient_embeddings, tsids = model.get_embeddings(test_dataloader)
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"retain_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"retain_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def BEHRTclassifier(sequences, 
                  labels,
                  word_to_idx=None, 
                  train_idx=None, 
                  valid_idx=None,
                  num_epochs=10,
                  batch_size=32,
                  embed_size=64,
                  hidden_size=128,
                  enable_plot=False,
                  frame_tqdm=None,
                  frame_plot=None,
                 ):
    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}

    train_dataset = BEHRTDataset(train_visists, labels_dict=labels, code2id=word_to_idx)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_dataset = BEHRTDataset(test_visists, code2id=word_to_idx)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = BEHRTModel(vocab_size=len(word_to_idx),embed_dim=embed_size,num_layers=2, num_heads=4, num_labels=2)

    model.train_model(train_dataloader, val_loader=None, num_epochs=num_epochs, lr=1e-4, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_dataloader)
    return y_pred, y_prob

def DipoleClassifier(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):

    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    train_dataset = DipoleDataset(train_visists, labels=labels, code2id=word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dipole_collate)
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}
    test_dataset = DipoleDataset(test_visists, labels=None, code2id=word_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=dipole_collate)
    
    model = DipoleModel(vocab_size=len(word_to_idx),embed_size=embed_size, hidden_size=hidden_size)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader)
    return y_pred, y_prob

def GRUclassifier(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):
    
    y_train = {id: val for (id,val) in labels.items() if id in train_idx} 
    y_valid = {id: val for (id,val) in labels.items() if id in valid_idx}
    train_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in train_idx}
    train_dataset = LSTMDataset(train_sentences, labels_dict=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)
    test_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in valid_idx}
    test_dataset = LSTMDataset(test_sentences, labels_dict=y_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)

    model = GRUModel(len(word_to_idx), embed_dim=embed_size, gru_hidden_dim=hidden_size, pooling="mean")
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader)
    return y_pred, y_prob

def GRUDclassifier(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):
    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    train_dataset = GRUDDataset(train_visists, labels_dict=labels, code2id=word_to_idx, max_seq_len=50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=grud_collate_fn)
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}
    test_dataset = GRUDDataset(test_visists, code2id=word_to_idx, max_seq_len=50)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=grud_collate_fn)

    model = GRUDModel(input_size=len(word_to_idx), 
                        hidden_size=hidden_size, output_size=2, 
                        x_mean=train_dataset.feature_means)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_dataloader)
    return y_pred, y_prob


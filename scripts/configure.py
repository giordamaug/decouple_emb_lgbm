from .embedding import FlexLSTMembedder, BEHRTembedder, COUNTEmbedder, TimeAwareLSTMEmbedder, DipoleEmbedder, CEHRBERTembedder
from .embedding import StaticEmbedder, RETAINembedder, DOMEEmbedder, BINARYEmbedder, GRUEmbedder, GRUEDembedder, Med2VecEmbedder
from .classifiers import FlexLSTMclassifier, DipoleClassifier, BEHRTclassifier, GRUclassifier, GRUDclassifier, Med2Vecclassifier

def configure(event_sequences, visit_sequences, event_sequences_type, labels, X_static, args):
    vocab = set()
    for patient_events in event_sequences.values():
        for event,_ in patient_events:
            vocab.update([event] if isinstance(event, str) else event)
    tmapper = {"comorbidity": 1, "infection": 2, "thrombosys": 3, "surgical_operation": 4, 
           "surgery": 4, "vaccination": 5, "therapy": 6, "platelet_change": 7, "bmi_change": 8, "followup": 9, "drug": 10} 
    word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}  # for LSTM, RETAIN, etc
    code2id = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    idx = 3
    for pid, visits in visit_sequences.items():
        for events, date in visits:
            for event in events:
                if event not in code2id:
                    code2id[event] = idx
                    idx += 1

    return { 
        "LSTM" : 
        {   "func": FlexLSTMembedder,
            "clf": FlexLSTMclassifier,
            "kwargs": {
                "name": "LSTM",
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot,
                "pooling": True,
                "use_padding": False,
                "bidirectional": False,
                "use_attention": False
            }
        },
        "BiPadLSTM" : 
        {   "func": FlexLSTMembedder,
            "clf": FlexLSTMclassifier,
            "kwargs": {
                "name": "BiPadLSTM",
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot,
                "pooling": True,
                "use_padding": True,
                "bidirectional": True,
                "use_attention": False
            }
        },
        "RETAIN" : 
        {   "func": RETAINembedder,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "CEHR-BERT" : 
        {   "func": CEHRBERTembedder,
            #"clf": BEHRTclassifier,
            "kwargs": {
                "sequences": event_sequences_type,
                "labels": labels,
                "tmapper": tmapper,
                "with_validation": True,
                "word_to_idx": code2id,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "BEHRT" : 
        {   "func": BEHRTembedder,
            "clf": BEHRTclassifier,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "Dipole" : 
        {   "func": DipoleEmbedder,
            "clf": DipoleClassifier,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "Med2Vec" : 
        {   "func": Med2VecEmbedder,
            "clf": Med2Vecclassifier,
                "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                'lambda_visit': 0.5,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "GRU-D" : 
        {   "func": GRUEDembedder,
            "clf": GRUDclassifier,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "GRU" : 
        {   "func": GRUEmbedder,
            "clf": GRUclassifier,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "embed_size": args.embedding_dim,
                "hidden_size": args.hidden_dim,
                "enable_plot": args.enable_plot
            }
        },
        "STATIC" : 
        {   "func": StaticEmbedder,
            "kwargs": {
                "df": X_static,
                "enable_plot": args.enable_plot
            }
        },
        "DOME" :
        {
            "func": DOMEEmbedder,
            "kwargs": {
                 "sequences": { id: events + [("dead", events[-1][1])] if labels[id]==1 else events for id,events in event_sequences.items()}, 
                 "targets": ["dead"],
                 "df": args.dataset,
                 "enable_plot": args.enable_plot
            }
        },
        "BINARY": 
        {
            "func": BINARYEmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "vocab": vocab,
                "targets": [args.target_var],
                "enable_plot": args.enable_plot
            }
        },
        "EVENT-CNT": 
        {
            "func": COUNTEmbedder,
            "kwargs": {
                "sequences": event_sequences_type,
                "enable_plot": args.enable_plot
            }
        }
    }

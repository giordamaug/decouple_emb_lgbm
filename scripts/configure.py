from .embedding import FlexLSTMembedder, BEHRTembedder, COUNTEREmbedder, TimeAwareLSTMEmbedder, DipoleEmbedder 
from .embedding import StaticEmbedder, RETAINembedder, DOMEEmbedder, BINARYEmbedder, GRUEmbedder, GRUEDembedder
from .classifiers import FlexLSTMclassifier, DipoleClassifier, BEHRTclassifier, GRUclassifier, GRUDclassifier

def configure(event_sequences, visit_sequences, labels, X_static, args):
    vocab = set()
    for patient_events in event_sequences.values():
        for event,_ in patient_events:
            vocab.update([event] if isinstance(event, str) else event)
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
        "tLSTM" : 
        {   "func": TimeAwareLSTMEmbedder,
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
        "COUNTER": 
        {
            "func": COUNTEREmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "vocab": vocab | set('dead'),
                "targets": [args.target_var],
                "enable_plot": args.enable_plot
            }
        }
    }

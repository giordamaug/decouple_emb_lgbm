import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import os
import random
import numpy as np
import torch
from tqdm.notebook import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

class Settings:
    def __init__(self, datafile = "datafile.json", 
                 pathology_field = '',
                 static_vars = [],
                 target_var = 'target',
                 pathologies = {}, 
                 methods = ["LSTM"], 
                 evfields = [],
                 no_selection = False,
                 spleen_flags = ['YES', 'NO'],
                 selected_spleen_flags = ['YES'],
                 remove_events = [],
                 enable_plot = True, to_latex=True, 
                 batch_size = 4, n_splits = 5, min_events = 3, hidden_size = 128, embedding_size=128, num_epochs=10,
                 lang='EN'):
        self.datafile = datafile
        self.num_epochs = num_epochs
        self.static_vars = static_vars
        self.target_var = target_var
        self.embedding_dim = embedding_size
        self.hidden_dim = hidden_size
        self.batch_size = batch_size
        self.pooling = 'mean'
        self.enable_plot = enable_plot
        self.n_splits = n_splits
        self.min_events = min_events
        self.random_state = 42
        self.to_latex = to_latex
        self.noselection = no_selection
        self.pathologies = pathologies
        self.methods = methods
        self.evfields = evfields
        self.remove_events= remove_events
        self.selected_patient_ids = []
        self.spleen_flags = spleen_flags
        self.selected_spleen_flags = selected_spleen_flags
        self.pathology_field=pathology_field
        self.events_field='events'
        self.is_splenectomized_field='is_splenectomized?'

        try:
            dataset = pd.read_json(self.datafile).set_index('id')
            self.dataset_orig = dataset
            self.dataset = self.dataset_orig.copy()

            if not self.noselection:
                self.dataset = self.dataset[(self.dataset[self.pathology_field].isin(self.pathologies.keys())) & 
                                            (self.dataset[self.is_splenectomized_field].isin(self.selected_spleen_flags))]
            
            print(f"Loaded {len(self.dataset)} records from {self.datafile}")
            self.selected_patient_ids = self.dataset.index.values
        except Exception as e:
            print("Error loading dataset:", e)


def pie_plot(dataset, pathology_field, pathologies):
    values = dataset[pathology_field].value_counts()
    legend_labels = values.index
    labels_short = [pathologies[l] for l in legend_labels]
    # 👉 funzione per mostrare valori assoluti
    def absolute_autopct(vals):
        def inner(pct):
            total = sum(vals)
            val = int(round(pct * total / 100.0))
            return f"{val}"
        return inner
    fig, ax = plt.subplots(figsize=(10, 3))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels_short,
        autopct=absolute_autopct(values)
    )
    # 👉 legenda separata
    ax.legend(
        wedges,
        legend_labels,
        title="Legend",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5)
    )
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------
# Sequence utility functions
#-------------------------------------------------------------------------------------

def truncevents(sequences, infection_list, max_inf=1, max_flwup=5, debug=False):
    trunc_sequences = {}
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_inf}"):
        inf_cnt = 0
        flw_cnt = 0
        new_evset = set()
        for e, d in sequences[id]:
              if e in infection_list:
                if debug: print(f"INF[{id}] {e}")
                new_evset.add((e,d))
                inf_cnt += 1
                if inf_cnt >= max_inf: break
              elif e == "followup" :
                new_evset.add((e,d))
                flw_cnt += 1
                if flw_cnt >= max_flwup: break
              else:
                if debug: print(f"eve[{id}] {e}")
                new_evset.add((e,d))
        trunc_sequences[id] = sorted(list(new_evset), key=lambda x: x[1])
    return trunc_sequences

def group_events_by_visit(sequences):
    visit_sequences= {}
    for pid, events in sequences.items():
        grouped_by_date = defaultdict(list)
        for event, date in events:
            grouped_by_date[date].append(event)
        visit_sequences[pid] = [(grouped_by_date[date], date) for date in sorted(grouped_by_date.keys())]
    return visit_sequences

def count_events_by_type(event_sequences):
    edf = pd.DataFrame(columns=['cardinality', 'n. instances', 'set'], 
                       index=pd.Series([], name='type'))
    for id, events in event_sequences.items():
        for event in events:
            if event[2] not in edf.index:
                row = pd.DataFrame([{'cardinality': 1, 'n. instances':1, 'set': set([event[0]])}],
                                    index=pd.Series([event[2]], name='type'))
                edf = pd.concat([edf, row], axis=0)
            else:
                edf.loc[event[2]]['set'].add(event[0])
                edf.loc[event[2], 'cardinality'] = len(edf.loc[event[2]]['set'])
                edf.loc[event[2], 'n. instances'] += 1
    return edf

def truncate_events_on1st_target(sequences, target_list, max_occurrence=1, field_name='event', debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_occurrence}"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[0] in target_list:
                    if debug: print(f"INF[{id}] {event[0]}")
                    inf_cnt += 1
                    if inf_cnt >= max_occurrence: 
                        y_df.loc[id] = 1
                        break
                    new_evset.append(event)
                else:
                    new_evset.append(event)
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

def truncate_events_on1st_infection(sequences, field_value='infection',  debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    y_df.index.name = "id"

    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[2] == field_value:
                    if debug: print(f"INF[{id}] {event[2]}")
                    inf_cnt += 1
                    if inf_cnt >= 1:
                        y_df.loc[id] = 1
                        break
                    new_evset.append(event[0:2])
                else:
                    new_evset.append(event[0:2])
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

def truncate_events_on1st_target_keepit(sequences, target_list, max_occurrence=1, field_name='event', debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_occurrence}"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[0] in target_list:
                    if debug: print(f"INF[{id}] {event[0]}")
                    new_evset.append(event)
                    inf_cnt += 1
                    if inf_cnt >= max_occurrence: 
                        y_df.loc[id] = 1
                        break
                else:
                    new_evset.append(event)
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

def cooccurring_to_target(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Cooccurrence removal"):
        # convert dates to datetype
        parsed_set = [(el, datetime.strptime(date_str, "%Y-%m-%d")) for el,date_str in sequences[id]]
        # Find most recent event,date pair
        if len(parsed_set) > 0:   # if the sequence is not null 
            # get dates form sequences
            _, dates = zip(*sequences[id])
            if len(set(dates)) > 1:                         # if at least two different dates
                max_date = max(date for _,date in parsed_set)
                # filter tuples with max date and with event in targets
                filtered_seq = [
                    (el, date.strftime("%Y-%m-%d"))
                    for el, date in parsed_set
                    if not (date == max_date and el not in targets + ['followup'])
                ]
                filtered_sequences[id] = filtered_seq
            else:
                seq = [
                    (el, date.strftime("%Y-%m-%d"))
                    for el, date in parsed_set
                ]
                filtered_sequences[id] = seq
        else:
            filtered_sequences[id] = parsed_set
    return filtered_sequences

def remove_target_from_sequences(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Target removal"):
        # filter tuples with event not in targets
        filtered_seq = [
                (el, date_str)
                for el,date_str in sequences[id]
                if el not in targets
        ]
        filtered_sequences[id] = filtered_seq
    return filtered_sequences

def remove_target_from_visit_sequences(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Target removal"):
        # filter tuples with event not in targets
        filtered_seq = [
                (list(set(els)-set(targets)), date_str)
                for els,date_str in sequences[id]
                if len(set(els)-set(targets)) > 0
        ]
        filtered_sequences[id] = filtered_seq
    return filtered_sequences

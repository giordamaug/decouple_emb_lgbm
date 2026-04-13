import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import os
import random
import numpy as np
import torch
from tqdm.notebook import tqdm
from collections import defaultdict

class Settings:
    def __init__(self, datafile = "datafile.json", 
                 patologies = [], 
                 methods = ["LSTM"], 
                 evfields = [],
                 no_selection = False,
                 spleen_flags = ['YES', 'NO'],
                 selected_spleen_flags = ['YES'],
                 enable_plot = True, with_static=False, with_validation=False, to_latex=True, 
                 batch_size = 4, n_splits = 5, min_events = 3, hidden_size = 128, embedding_size=128, num_epochs=10,
                 lang='EN'):
        self.selected_patient_ids = []
        self.data_file = datafile
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_size
        self.hidden_dim = hidden_size
        self.batch_size = batch_size
        self.pooling = 'mean'
        self.enable_plot = enable_plot
        self.with_static = with_static
        self.with_validation = with_validation
        self.n_splits = n_splits
        self.min_events = min_events
        self.random_state = 42
        self.to_latex = to_latex
        self.noselection = no_selection
        self.patologies = patologies
        self.methods = methods
        self.dataset = None
        self.dataset_orig = None
        self.evfields = evfields
        self.selected_patient_ids = []
        self.spleen_flags = spleen_flags
        self.lang = lang
        self.selected_spleen_flags = selected_spleen_flags
        self.spleen_yes_flag = spleen_flags[0]
        self.spleen_no_flag = spleen_flags[1]
        if lang == 'EN':
            self.yes_answer = 'YES',
            self.no_answer = 'NO',
            self.pathology_field='base_pathology_area'
            self.pathology_field='base_pathology_area'
            self.events_field='events'
            self.is_splenectomized_field='is_splenectomized?'
        elif lang =='IT':
            self.yes_answer = 'SI',
            self.no_answer = 'NO',
            self.pathology_field='area_pat_base'
            self.events_field='eventi'
            self.is_splenectomized_field='splenectomizzato'
        self.splenectomized = self.yes_answer

        
class SettingsWidget:
    def __init__(self, args):
        self.args = args

        self.data_file_widget = widgets.Text(
            value=args.data_file,
            description='Data file:',
            layout=widgets.Layout(width='400px')
        )

        self.load_output = widgets.Output()
        
        self.num_epochs_widget = widgets.IntText(
            value=args.num_epochs,
            description='Num epochs:'
        )
        
        self.embedding_dim_widget = widgets.IntText(
            value=args.embedding_dim,
            description='Embedding dim:'
        )
        
        self.hidden_dim_widget = widgets.IntText(
            value=args.hidden_dim,
            description='Hidden dim:'
        )
        
        self.batch_size_widget = widgets.IntText(
            value=args.batch_size,
            description='Batch size:'
        )
        
        self.pooling_widget = widgets.Dropdown(
            options=['mean', 'max', 'sum'],
            value=args.pooling,
            description='Pooling:'
        )
        
        self.enable_plot_widget = widgets.Checkbox(
            value=args.enable_plot,
            description='Enable plot'
        )
        
        self.with_static_widget = widgets.Checkbox(
            value=args.with_static,
            description='With static feats'
        )

        self.with_validation_widget = widgets.Checkbox(
            value=args.with_validation,
            description='With validation'
        )

        self.n_splits_widget = widgets.IntText(
            value=args.n_splits,
            description='N splits:'
        )
        
        self.min_events_widget = widgets.IntText(
            value=args.min_events,
            description="Min Events:"
        )

        self.random_state_widget = widgets.IntText(
            value=args.random_state,
            description='Random state:'
        )
        
        # Patologies widget - initially empty
        self.patologies_widget = widgets.SelectMultiple(
            options=[],
            description='Patologies:',
            rows=7,
            layout=widgets.Layout(width='500px')
        ) if not self.args.noselection else widgets.FloatText()

        # Patologies widget - initially empty
        self.methods_widget = widgets.SelectMultiple(
            value=args.methods,
            options=args.methods,
            description='Methods:',
            rows=4,
            layout=widgets.Layout(width='evfields500px')
        )

        # Event field widget - initially empty
        self.evfields_widget = widgets.SelectMultiple(
            #value=args.evfields,
            #options=[],
            description='Event Fields:',
            rows=4,
            layout=widgets.Layout(width='500px')
        )
        
        # Event field widget - initially empty
        self.splenectomized_widget = widgets.SelectMultiple(
            value=self.args.selected_spleen_flags,
            options=[self.args.spleen_yes_flag, self.args.spleen_no_flag],
            description='Splenectomized?:',
            rows=2,
            layout=widgets.Layout(width='500px')
        ) if not self.args.noselection else widgets.FloatText()

        #self.splenectomized_widget = widgets.Checkbox(
        #    value=(args.splenectomized == self.args.yes_answer),
        #    description='Splenectomized?'
        #) if not self.args.noselection else widgets.FloatText()
        
        self.update_output = widgets.Output()

        # Add observers for auto-update
        for w in [
            self.data_file_widget,
            self.num_epochs_widget,
            self.embedding_dim_widget,
            self.hidden_dim_widget,
            self.batch_size_widget,
            self.pooling_widget,
            self.enable_plot_widget,
            self.with_static_widget,
            self.with_validation_widget,
            self.n_splits_widget,
            self.min_events_widget,
            self.random_state_widget,
            self.patologies_widget,
            self.methods_widget,
            self.evfields_widget,
            self.splenectomized_widget
        ]:
            w.observe(self.on_update_clicked, names='value')

        # Add observer for data_file
        self.data_file_widget.observe(self.on_load_dataset, names='value')
        self.on_load_dataset({'name':'value', 'new': self.data_file_widget.value})

    def display(self):
        display(
            widgets.VBox([
                self.data_file_widget,
                self.load_output,
                self.num_epochs_widget,
                self.embedding_dim_widget,
                self.hidden_dim_widget,
                self.batch_size_widget,
                self.pooling_widget,
                self.enable_plot_widget,
                self.with_static_widget,
                self.with_validation_widget,
                self.n_splits_widget,
                self.min_events_widget,
                self.random_state_widget,
                self.patologies_widget,
                self.methods_widget,
                self.evfields_widget,
                self.splenectomized_widget,
                self.update_output
            ])
        )
    
    def on_load_dataset(self, change):
        with self.load_output:
            clear_output()
            try:
                file_path = self.data_file_widget.value
                dataset = pd.read_json(file_path).set_index('id')
                self.args.dataset_orig = dataset
                self.args.dataset = self.args.dataset_orig.copy()

                if not self.args.noselection:
                    pathology_values = sorted(
                        dataset[self.args.pathology_field].dropna().unique().tolist()
                    )
                
                    self.patologies_widget.options = [(p, p) for p in pathology_values]
                    self.patologies_widget.value = pathology_values if len(self.args.patologies) == 0 else self.args.patologies
                
                evfvalues = list(dataset.iloc[0][self.args.events_field][0].keys())
                self.evfields_widget.options = [(p, p) for p in evfvalues] 
                self.evfields_widget.value = evfvalues if len(self.args.evfields) == 0 else self.args.evfields
                print(f"Dataset loaded successfully from {file_path}")
            except Exception as e:
                print("Error loading dataset:", e)
    
    def on_update_clicked(self, change=None):
        with self.update_output:
            clear_output()
            
            self.args.data_file = self.data_file_widget.value
            self.args.num_epochs = self.num_epochs_widget.value
            self.args.embedding_dim = self.embedding_dim_widget.value
            self.args.hidden_dim = self.hidden_dim_widget.value
            self.args.batch_size = self.batch_size_widget.value
            self.args.pooling = self.pooling_widget.value
            self.args.enable_plot = self.enable_plot_widget.value
            self.args.with_static = self.with_static_widget.value
            self.args.with_validation = self.with_validation_widget.value
            self.args.n_splits = self.n_splits_widget.value
            self.args.min_events = self.min_events_widget.value
            self.args.random_state = self.random_state_widget.value
            self.args.patologies = list(self.patologies_widget.value) if not self.args.noselection else []
            self.args.methods= list(self.methods_widget.value)
            #self.args.splenectomized = self.args.yes_answer if self.splenectomized_widget.value else "NO"
            self.args.splenectomized = self.splenectomized_widget.value
            self.args.random_state = self.random_state_widget.value
            if len(self.evfields_widget.value) > 0:
                self.args.evfields = self.evfields_widget.value
            # Imposta il seme di tutti i moduli principali
            os.environ['PYTHONHASHSEED'] = str(self.args.random_state)
            random.seed(self.args.random_state)
            np.random.seed(self.args.random_state)
            torch.manual_seed(self.args.random_state)
            torch.cuda.manual_seed_all(self.args.random_state)

            if self.args.dataset is not None:
                if self.args.noselection:
                    self.args.selected_patient_ids = self.args.dataset.index.values
                else:
                    try:
                        self.args.selected_patient_ids = self.args.dataset_orig[
                            (self.args.dataset_orig[self.args.pathology_field].isin(self.args.patologies)) & 
                            #(self.args.dataset_orig[self.args.is_splenectomized_field] == self.args.splenectomized)
                            (self.args.dataset_orig[self.args.is_splenectomized_field].isin(self.args.splenectomized))
                        ].index.values
                    except Exception:
                        self.args.selected_patient_ids = []
                self.args.dataset = self.args.dataset_orig.copy().loc[self.args.selected_patient_ids]
            else:
                self.args.selected_patient_ids = []

            print("Settings updated:")
            for attr in vars(self.args):
                if attr == "dataset_orig":
                    continue
                elif attr == "dataset":
                    if self.args.dataset is not None:
                        print(f"{attr}: loaded DataFrame with shape {self.args.dataset.shape}")
                    else:
                        print(f"{attr}: None")
                elif attr == "selected_patient_ids":
                    print(f"{attr}: {len(self.args.selected_patient_ids)} patients")
                else:
                    print(f"{attr}: {getattr(self.args, attr)}")
                            
    def get_settings(self):
        return self.args

def df_to_latex_bold(df, float_format="%.3f", with_cm=False):
    """
    Rende LaTeX da un DataFrame con metodi come indici e metriche come colonne.
    Evidenzia in grassetto i massimi per ogni metrica numerica.
    """
    df_copy = df.copy().astype(object)

    for col in df.columns:
        if col == "CM" and with_cm:
            # Riformatta la confusion matrix in due righe
            new_cm = []
            for cm in df[col]:
                try:
                    formatted = (
                        f"\cm{{{cm[0][0]}}}{{{cm[0][1]}}}{{{cm[1][0]}}}{{{cm[1][1]}}}{{A}}{{D}}{{{cm[0][0]+cm[0][1]}}}{{{cm[1][0]+cm[1][1]}}}"
                    )
                except Exception:
                    formatted = str(cm)
                new_cm.append(formatted)
            df_copy[col] = new_cm
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                if col.endswith(' mean'):
                    colname = col.rstrip(' mean')
                    max_val = df[col].max()
                    for idx in df.index:
                        val = df.at[idx, col]
                        formatted = f"{float_format % val}$\\pm${float_format % df.at[idx, colname+' std']}"
                        if val == max_val:
                            formatted = f"\\textbf{{{formatted}}}"
                        df_copy.at[idx, col] = formatted
                    df_copy.drop(colname+' std', axis=1, inplace=True)
                    df_copy.rename(columns={col: colname}, inplace=True)
            else:
                # Mantiene valori non numerici (come matrici di confusione)
                df_copy[col] = df_copy[col].astype(str)
    return df_copy.to_latex(escape=False)

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

def group_events_by_visit_old(sequences, on_field='date'):
    visit_sequences= {}
    for pid, events in sequences.items():
        grouped_by_date = defaultdict(list)
        for event in events:
            date = event[on_field]
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
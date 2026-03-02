import io
import json
import logging
import textwrap
import sys
from pathlib import Path
from getpass import getpass
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from itertools import combinations

# Data Science & Visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# UI & Notebook Display
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown, clear_output

# Specialized ML & API Tools
import argilla as rg
import requests
from google import genai
from google.colab import drive
from gliner2 import GLiNER2


def generate_adjudication_ui(ready_data):
    if not ready_data:
        print("No multi-annotated records found.")
        return

    # 1. Filter for records with any form of disagreement
    disputed_records = []
    for item in ready_data:
        users = list(item["annotations"].keys())
        if len(users) >= 2:
            u1, u2 = users[0], users[1]
            if set(item["annotations"][u1]) != set(item["annotations"][u2]):
                disputed_records.append(item)

    if not disputed_records:
        print("🎉 Perfect consensus across all records! Nothing to adjudicate.")
        return

    # 2. UI Components
    slider = widgets.IntSlider(
        min=0, max=len(disputed_records)-1, 
        description='Record:', continuous_update=False,
        layout=widgets.Layout(width='400px')
    )
    
    btn_prev = widgets.Button(description="← Previous", layout=widgets.Layout(width='100px'))
    btn_next = widgets.Button(description="Next →", layout=widgets.Layout(width='100px'))
    
    output = widgets.Output()

    # 3. Navigation Logic
    def on_next_clicked(b):
        if slider.value < slider.max:
            slider.value += 1

    def on_prev_clicked(b):
        if slider.value > slider.min:
            slider.value -= 1

    btn_next.on_click(on_next_clicked)
    btn_prev.on_click(on_prev_clicked)

    def render_view(index):
        with output:
            clear_output(wait=True)
            record = disputed_records[index]
            users = list(record["annotations"].keys())
            u1, u2 = users[0], users[1]
            
            text = record["text"]
            spans1 = set(record["annotations"][u1])
            spans2 = set(record["annotations"][u2])
            
            # Calculate Diffs
            only_u1 = spans1 - spans2
            only_u2 = spans2 - spans1

            def highlight_text(base_text, diff_spans, color):
                html_snippet = base_text
                for s, e, l in sorted(list(diff_spans), key=lambda x: x[0], reverse=True):
                    tag = f"<span style='background-color:{color}; border-bottom:2px solid black; padding:0 2px;'><b>{html_snippet[s:e]}</b> <small>({l})</small></span>"
                    html_snippet = html_snippet[:s] + tag + html_snippet[e:]
                return html_snippet

            # Rendering
            display(HTML(f"<h2>Adjudication: {record['id']} <small>({index+1}/{len(disputed_records)})</small></h2>"))
            
            display(HTML("<h3>Visual Diffs (Only differences shown)</h3>"))
            display(HTML(f"<div style='line-height:2.0; border:1px solid #ccc; padding:15px; border-radius:8px; background:#f9f9f9;'>"))
            display(HTML(f"<b>Only {u1} (Red):</b> {highlight_text(text, only_u1, '#ffcccc')}"))
            display(HTML("<hr>"))
            display(HTML(f"<b>Only {u2} (Blue):</b> {highlight_text(text, only_u2, '#cce5ff')}"))
            display(HTML("</div>"))

            display(HTML("<h3>Full Annotation Sets</h3>"))
            table_html = "<table style='width:100%; border-collapse:collapse;'><tr>"
            for user in [u1, u2]:
                table_html += f"<th style='border:1px solid #ddd; padding:8px; background:#eee;'>{user} Annotations</th>"
            table_html += "</tr><tr>"
            
            for user in [u1, u2]:
                anno_list = sorted(record["annotations"][user], key=lambda x: x[0])
                list_items = "".join([f"<li>[{s}-{e}] <b>{l}</b>: {text[s:e]}</li>" for s, e, l in anno_list])
                table_html += f"<td style='border:1px solid #ddd; padding:8px; vertical-align:top;'><ul>{list_items}</ul></td>"
            
            table_html += "</tr></table>"
            display(HTML(table_html))

    # 4. Interactivity
    def on_slider_change(change):
        render_view(change['new'])

    slider.observe(on_slider_change, names='value')
    
    # Layout: Put buttons and slider in a row
    nav_controls = widgets.HBox([btn_prev, slider, btn_next])
    
    display(nav_controls)
    display(output)
    render_view(0)


# --- 2. DATA EXTRACTION & PREPARATION ---
def get_ready_data(dataset) -> List[Dict[str, Any]]:
    # Fetch completed records with responses
    query_completed = rg.Query(filter=rg.Filter(("status", "==", "completed")))
    completed_records = list(dataset.records(query=query_completed, with_responses=True))
    
    # Map UUIDs to Usernames
    user_id_to_name = {str(u.id): u.username for u in client.users}
    
    formatted_list = []
    for record in completed_records:
        user_responses = {}
        for resp in record.responses:
            if resp.status == "submitted" and resp.value is not None:
                name = user_id_to_name.get(str(resp.user_id), str(resp.user_id))
                # Standardize to (start, end, label)
                user_responses[name] = [(s['start'], s['end'], s['label']) for s in resp.value]
        
        formatted_list.append({
            "id": record.metadata.get("original_id", "unknown"),
            "domain": record.metadata.get("domain_filter") or record.metadata.get("domain") or "Unknown Domain",
            "text": record.fields.get("text", ""),
            "annotations": user_responses 
        })
    
    # Filter for records with 2+ annotators
    return [item for item in formatted_list if len(item["annotations"]) >= 2]

# --- 3. METRIC FUNCTIONS ---
def calculate_pair_metrics(iaa_ready: List[Dict]) -> pd.DataFrame:
    pairs = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "matches": 0, "total": 0})
    for item in iaa_ready:
        users = sorted(list(item["annotations"].keys()))
        for u1, u2 in combinations(users, 2):
            pair_key = (u1, u2)
            s1, s2 = set(item["annotations"][u1]), set(item["annotations"][u2])
            pairs[pair_key]["total"] += 1
            if s1 == s2: pairs[pair_key]["matches"] += 1
            pairs[pair_key]["tp"] += len(s1 & s2)
            pairs[pair_key]["fp"] += len(s1 - s2)
            pairs[pair_key]["fn"] += len(s2 - s1)
    
    res = []
    for (u1, u2), c in pairs.items():
        f1 = (2*c["tp"])/(2*c["tp"]+c["fp"]+c["fn"]) if (2*c["tp"]+c["fp"]+c["fn"]) > 0 else 0
        res.append({
            "Annotator Pair": f"{u1} ↔ {u2}",
            "Sentences": c["total"],
            "Perfect Match %": f"{(c['matches']/c['total']):.1%}",
            "Span F1": round(f1, 4)
        })
    return pd.DataFrame(res)

def calculate_label_metrics(iaa_ready: List[Dict], target_pair: List[str]) -> pd.DataFrame:
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    u1, u2 = target_pair[0], target_pair[1]
    
    for item in iaa_ready:
        annos = item["annotations"]
        if u1 in annos and u2 in annos:
            s1, s2 = set(annos[u1]), set(annos[u2])
            all_labels = {s[2] for s in s1} | {s[2] for s in s2}
            for label in all_labels:
                l1, l2 = {s for s in s1 if s[2] == label}, {s for s in s2 if s[2] == label}
                stats[label]["tp"] += len(l1 & l2)
                stats[label]["fp"] += len(l1 - l2)
                stats[label]["fn"] += len(l2 - l1)
    
    res = []
    for label, c in stats.items():
        f1 = (2*c["tp"])/(2*c["tp"]+c["fp"]+c["fn"]) if (2*c["tp"]+c["fp"]+c["fn"]) > 0 else 0
        res.append({"Label": label, "Both Agreed": c["tp"], f"Only {u1}": c["fp"], f"Only {u2}": c["fn"], "F1-Score": round(f1, 4)})
    return pd.DataFrame(res).sort_values("F1-Score", ascending=False)

# --- 4. VISUALIZATION FUNCTIONS ---
def plot_domain_confusion_matrix(ready_data: List[Dict], u1: str, u2: str):
    y_true, y_pred = [], []
    domain_name = "Unknown Domain"
    
    for item in ready_data:
        annos = item.get("annotations", {})
        if u1 in annos and u2 in annos:
            if domain_name == "Unknown Domain":
                domain_name = item.get("domain", "Unknown Domain")
            off1 = {(s, e): l for s, e, l in annos[u1]}
            off2 = {(s, e): l for s, e, l in annos[u2]}
            common = set(off1.keys()) & set(off2.keys())
            for o in common:
                y_true.append(off1[o])
                y_pred.append(off2[o])
                
    if not y_true: return

    labels = sorted(list(set(y_true) | set(y_pred)))
    df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=.5, annot_kws={"size": 9})
    plt.title(f"Domain: {domain_name}\n({u1} vs {u2})", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def load_gsheet_data(texts_url: str, labels_url: str, merge: bool = False):
    """
    Downloads Google Sheets as CSVs and returns them as DataFrames.
    
    Args:
        texts_url: The sharing URL for the texts sheet.
        labels_url: The sharing URL for the labels sheet.
        merge: If True, returns a single DataFrame joined on 'domain'.
    """
    def get_csv_url(url: str):
        try:
            file_id = url.split("/")[-2]
            return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
        except IndexError:
            raise ValueError(f"Invalid Google Sheets URL: {url}")

    try:
        # Load datasets
        df_texts = pd.read_csv(get_csv_url(texts_url))
        df_labels = pd.read_csv(get_csv_url(labels_url))
        
        logging.info("✅ DataFrames loaded successfully.")
        
        if merge:
            # Joining on 'domain' as per your column structure
            df_combined = pd.merge(df_texts, df_labels, on="domain", how="left")
            return df_combined
        
        return df_texts, df_labels

    except Exception as e:
        logging.error(f"❌ Loading failed: {e}")
        return None


def display_styled_df(df: pd.DataFrame, target_col: str = 'text', n_samples: int = 10):
    """
    Applies custom CSS and Pandas styling to display a readable, wrapped DataFrame.
    Optimized for long-form text (like Greek) alongside metadata.
    """
    # 1. Inject CSS for Jupyter/Colab output wrapping
    display(HTML("<style>.output_subarea { white-space: normal !important; }</style>"))

    # 2. Reset Pandas truncation
    pd.set_option('display.max_colwidth', None)

    # 3. Handle sampling logic
    sample_size = min(n_samples, len(df))
    df_sample = df.sample(n=sample_size)

    # 4. Define and apply styles
    other_cols = df_sample.columns.difference([target_col])
    
    styled = (df_sample.style
        .set_properties(
            subset=[target_col], 
            **{
                'text-align': 'left',
                'white-space': 'normal',
                'min-width': '500px',
                'font-family': 'serif',
                'background-color': '#f9f9f9' # Subtle highlight for the text
            }
        )
        .set_properties(
            subset=other_cols,
            **{
                'text-align': 'center',
                'white-space': 'nowrap',
                'min-width': '100px',
                'vertical-align': 'middle'
            }
        )
    )

    display(styled)


def prepare_gliner_dataset(df_texts: pd.DataFrame, df_labels: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Combines text data with domain-specific labels and definitions.
    Returns a list of structured records ready for GLiNER2 inference.
    """
    # 1. Pre-compute mappings for performance
    # domain -> [list of labels]
    domain_labels = df_labels.groupby('domain')['label'].apply(list).to_dict()
    
    # label -> definition string
    label_definitions = df_labels.set_index('label')['definition'].to_dict()

    logging.info(f"🔍 Mapping {len(domain_labels)} domains with {len(label_definitions)} unique entity definitions.")

    dataset_raw = []

    # 2. Iterate and structure
    for _, row in df_texts.iterrows():
        domain = row.get('domain')
        
        # Only process if we have defined labels for this domain
        if domain in domain_labels:
            labels_for_domain = domain_labels[domain]
            
            record = {
                "id": int(row['id']),
                "domain": domain,
                "text": row['text'],
                "starter_labels": labels_for_domain,
                "definitions": {
                    lbl: label_definitions.get(lbl, "No definition provided") 
                    for lbl in labels_for_domain
                }
            }
            dataset_raw.append(record)
        else:
            logging.warning(f"⚠️ Skipping ID {row.get('id')}: Domain '{domain}' not found in labels sheet.")

    logging.info(f"🚀 Prepared {len(dataset_raw)} records for GLiNER2 inference.")
    return dataset_raw

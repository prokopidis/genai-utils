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

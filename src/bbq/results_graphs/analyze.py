#!/usr/bin/env python3
"""BBQ Experiment Analysis

Minimal notebook for BBQ experiment visualization.
Switch models by changing MODEL, then run cells.

Usage:
    Run cells interactively with #%% markers in VS Code/Cursor
"""

#%% Configuration & Setup (run first)
MODEL = '8B'  # Switch: '1.7B', '8B', '32B'
SAVE = False

import os
import sys
from pathlib import Path

_script_dir = Path("/workspace/experiments/exp_006_extend_thinking/bbq/results_graphs")
os.chdir(_script_dir)
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from analysis_utils import (
    load_data,
    cross_model_summary_table,
    category_breakdown_table,
    model_summary_table,
)

# Load data once
data = load_data()
print("Data loaded for models:", list(data.keys()))

#%% Cross-Model Summary Table (all models side-by-side)
cross_model_summary_table(data, save=SAVE)

#%% Category Breakdown (single model)
category_breakdown_table(data, model=MODEL, save=SAVE)

#%% Single-Model Summary Table
model_summary_table(data, model=MODEL, save=SAVE)

# %%

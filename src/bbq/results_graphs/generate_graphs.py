#!/usr/bin/env python3
"""Generate BBQ graphs for MATS write-up.

Usage:
    python generate_graphs.py
    
Or run cells interactively with #%% markers in VS Code/Cursor.
"""

#%% Imports
import os
import sys
from pathlib import Path
_script_dir = Path("/workspace/experiments/exp_006_extend_thinking/bbq/results_graphs")
os.chdir(_script_dir)
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from graph_utils import (
    plot_blank_static_accuracy, 
    plot_incorrect_static_accuracy, 
    plot_summary_table,
    plot_mats_summary_table,
)

#%% MATS Summary Table - All Models (Cross-Model Comparison)
# This generates the table with:
# Rows: CoT, NoCoT, Blank Best, Blank Worst, Incorrect Best, Incorrect Worst
# Columns: Experiment | 1.7B Q% | 1.7B S% | 8B Q% | 8B S% | 32B Q% | 32B S%
plot_mats_summary_table(save=True)

#%% 8B Model - All Categories
#plot_blank_static_accuracy(model='32B', save=True)
plot_summary_table(model='32B', save=False)

#%% 8B Model - Age & Appearance Only
plot_blank_static_accuracy(model='32B', save=False)
#plot_summary_table(model='8B', categories=['age', 'appearance'], save=True)

#%% 32B Model - All Categories
plot_blank_static_accuracy(model='32B', save=True)
plot_summary_table(model='32B', save=True)

# %%
# Incorrect answer graphs
plot_incorrect_static_accuracy(model='8B', save=True)
plot_incorrect_static_accuracy(model='8B', categories=['age', 'appearance'], save=True)
plot_incorrect_static_accuracy(model='32B', save=True)
# %%

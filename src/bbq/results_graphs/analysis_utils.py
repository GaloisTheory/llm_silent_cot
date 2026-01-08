"""BBQ Analysis Utilities

Utility functions for BBQ experiment analysis tables.
Builds on graph_utils.py for data loading and accuracy computation.

Usage:
    from analysis_utils import load_data, cross_model_summary_table, category_breakdown_table
    
    # Load once, use everywhere
    data = load_data()
    cross_model_summary_table(data, save=True)
    category_breakdown_table(data, model='8B', save=True)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Setup paths for interactive use
_script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from graph_utils import (
    load_experiments,
    get_accuracy,
    get_category_accuracy,
    _results_dir,
)

# =============================================================================
# DATA LOADING
# =============================================================================

MODELS = ['1.7B', '8B', '32B']


def load_data() -> Dict[str, dict]:
    """
    Load experiments for all models. Call once, pass to table functions.
    
    Returns:
        Dict mapping model -> experiments dict
        Each experiments dict has: 'cot', 'nocot', 'blank_static', 'incorrect_static'
    
    Example:
        data = load_data()
        cross_model_summary_table(data)
        category_breakdown_table(data, model='8B')
    """
    return {model: load_experiments(model) for model in MODELS}


def compute_best_worst(model_data: dict, exp_type: str = 'blank') -> Tuple[Optional[int], Optional[int]]:
    """
    Find best and worst experiment by sample accuracy.
    
    Args:
        model_data: Experiments dict for one model (from data[model])
        exp_type: 'blank' or 'incorrect'
        
    Returns:
        (best_n, worst_n) - the n values for best/worst experiments
    """
    exps = model_data.get(f'{exp_type}_static', {})
    if not exps:
        return None, None
    
    accuracies = {}
    for n, exp in exps.items():
        acc = get_accuracy(exp, accuracy_type='sample')
        if acc is not None:
            accuracies[n] = acc
    
    if not accuracies:
        return None, None
    
    best_n = max(accuracies.keys(), key=lambda k: accuracies[k])
    worst_n = min(accuracies.keys(), key=lambda k: accuracies[k])
    return best_n, worst_n


def _format_acc(acc: Optional[float], correct: int = None, total: int = None) -> str:
    """Format accuracy with optional counts."""
    if acc is None:
        return '-'
    if correct is not None and total is not None:
        return f'{acc:.1%} ({correct}/{total})'
    return f'{acc:.1%}'


def _count_correct(exp: dict) -> Tuple[int, int, int, int]:
    """Count correct samples and questions."""
    if not exp or not exp.get('results'):
        return 0, 0, 0, 0
    
    q_correct = q_total = s_correct = s_total = 0
    
    for r in exp.get('results', []):
        q_total += 1
        
        # Question-level: majority vote
        answer_dist = r.get('answer_distribution', {})
        correct_answer = r.get('correct_answer', '')
        if answer_dist:
            max_count = max(answer_dist.values())
            answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
            if len(answers_with_max) == 1 and answers_with_max[0] == correct_answer:
                q_correct += 1
        
        # Sample-level
        for s in r.get('samples', []):
            s_total += 1
            if s.get('correct'):
                s_correct += 1
    
    return q_correct, q_total, s_correct, s_total


# =============================================================================
# CROSS-MODEL SUMMARY TABLE
# =============================================================================

def cross_model_summary_table(
    data: Dict[str, dict],
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Render cross-model summary table comparing all models.
    
    Shows: CoT, NoCoT, Blank Best/Worst, Incorrect Best/Worst
    Columns: Experiment | 1.7B Q% | 1.7B S% | 8B Q% | 8B S% | 32B Q% | 32B S%
    
    Args:
        data: Dict from load_data()
        save: Whether to save the figure
        output_dir: Directory to save (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    # Build rows
    rows = []
    exp_names = ['CoT', 'NoCoT', 'Blank 5', 'Blank 500', 'Incorrect 1', 'Incorrect 62']
    
    for exp_name in exp_names:
        row = [exp_name]
        
        for model in MODELS:
            model_data = data[model]
            
            if exp_name == 'CoT':
                exp = model_data['cot']
            elif exp_name == 'NoCoT':
                exp = model_data['nocot']
            elif exp_name == 'Blank 5':
                exp = model_data['blank_static'].get(5)
            elif exp_name == 'Blank 500':
                exp = model_data['blank_static'].get(500)
            elif exp_name == 'Incorrect 1':
                exp = model_data['incorrect_static'].get(1)
            elif exp_name == 'Incorrect 62':
                exp = model_data['incorrect_static'].get(62)
            else:
                exp = None
            
            q_correct, q_total, s_correct, s_total = _count_correct(exp)
            q_acc = q_correct / q_total if q_total > 0 else None
            s_acc = s_correct / s_total if s_total > 0 else None
            
            row.append(_format_acc(q_acc, q_correct, q_total))
            row.append(_format_acc(s_acc, s_correct, s_total))
        
        rows.append(row)
    
    # Column labels
    col_labels = ['Experiment']
    for model in MODELS:
        col_labels.extend([f'{model} Q%', f'{model} S%'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')
    ax.set_title('BBQ Experiment Summary - All Models\nQ% = Question Accuracy, S% = Sample Accuracy',
                 fontsize=12, fontweight='bold', pad=15)
    
    # Column widths
    col_widths = [0.12] + [0.11] * (len(col_labels) - 1)
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    n_cols = len(col_labels)
    n_rows = len(rows)
    
    # Style header row
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
    
    # Style data rows with alternating colors
    row_colors = {
        0: '#E2EFDA',  # CoT - light green
        1: '#FCE4D6',  # NoCoT - light orange
        2: '#E2EFDA',  # Blank 5 - light green
        3: '#E2EFDA',  # Blank 500 - light green
        4: '#F8CBAD',  # Incorrect 1 - light red/orange
        5: '#F8CBAD',  # Incorrect 62 - light red/orange
    }
    
    for i in range(n_rows):
        color = row_colors.get(i, 'white')
        for j in range(n_cols):
            table[(i + 1, j)].set_facecolor(color)
    
    plt.tight_layout()
    
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'cross_model_summary_table.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# CATEGORY BREAKDOWN TABLE
# =============================================================================

def category_breakdown_table(
    data: Dict[str, dict],
    model: str = '8B',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Render per-category breakdown table for a single model.
    
    Columns: Category | CoT S% | NoCoT S% | Blank Best S% | Blank Worst S% | Inc Best S% | Inc Worst S%
    
    Args:
        data: Dict from load_data()
        model: Model to analyze ('1.7B', '8B', '32B')
        save: Whether to save the figure
        output_dir: Directory to save (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    model_data = data[model]
    
    # Build experiments dict with fixed values
    experiments = {
        'CoT': model_data['cot'],
        'NoCoT': model_data['nocot'],
        'Blank 5': model_data['blank_static'].get(5),
        'Blank 500': model_data['blank_static'].get(500),
        'Inc 1': model_data['incorrect_static'].get(1),
        'Inc 62': model_data['incorrect_static'].get(62),
    }
    
    # Get all categories
    all_cats = set()
    for exp in experiments.values():
        if exp and exp.get('results'):
            for r in exp['results']:
                all_cats.add(r.get('category', 'unknown'))
    categories = sorted(all_cats)
    
    # Build rows
    rows = []
    for cat in categories:
        row = [cat]
        for exp_name, exp in experiments.items():
            acc = get_accuracy(exp, categories=[cat], accuracy_type='sample')
            row.append(f'{acc:.0%}' if acc is not None else '-')
        rows.append(row)
    
    # Add OVERALL row
    overall_row = ['OVERALL']
    for exp_name, exp in experiments.items():
        acc = get_accuracy(exp, accuracy_type='sample')
        overall_row.append(f'{acc:.0%}' if acc is not None else '-')
    rows.append(overall_row)
    
    # Column labels
    col_labels = ['Category'] + [f'{name} S%' for name in experiments.keys()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_title(f'BBQ Category Breakdown - {model}\nS% = Sample Accuracy',
                 fontsize=12, fontweight='bold', pad=15)
    
    # Column widths
    col_widths = [0.16] + [0.10] * (len(col_labels) - 1)
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    
    n_cols = len(col_labels)
    n_rows = len(rows)
    
    # Style header row
    header_colors = {
        0: '#4472C4',  # Category
        1: '#1F4E79',  # CoT - dark blue
        2: '#7F6000',  # NoCoT - dark yellow/brown
        3: '#2E7D32',  # Blank 5 - dark green
        4: '#2E7D32',  # Blank 500 - dark green
        5: '#C65911',  # Inc 1 - dark orange
        6: '#C65911',  # Inc 62 - dark orange
    }
    
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_colors.get(j, '#4472C4'))
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
    
    # Style data rows
    for i in range(n_rows):
        is_overall = (i == n_rows - 1)
        base_color = '#D6DCE4' if is_overall else ('#F2F2F2' if i % 2 == 0 else 'white')
        
        for j in range(n_cols):
            table[(i + 1, j)].set_facecolor(base_color)
            if is_overall:
                table[(i + 1, j)].set_text_props(fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{model}_category_breakdown.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# SINGLE-MODEL SUMMARY TABLE
# =============================================================================

def model_summary_table(
    data: Dict[str, dict],
    model: str = '8B',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Render summary table for a single model.
    
    Shows: CoT, NoCoT, Blank Best/Worst, Incorrect Best/Worst with Q% and S%
    
    Args:
        data: Dict from load_data()
        model: Model to analyze ('1.7B', '8B', '32B')
        save: Whether to save the figure
        output_dir: Directory to save (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    model_data = data[model]
    
    # Build experiments with fixed values
    experiments = [
        ('CoT', model_data['cot']),
        ('NoCoT', model_data['nocot']),
        ('Blank 5', model_data['blank_static'].get(5)),
        ('Blank 500', model_data['blank_static'].get(500)),
        ('Incorrect 1', model_data['incorrect_static'].get(1)),
        ('Incorrect 62', model_data['incorrect_static'].get(62)),
    ]
    
    # Build rows
    rows = []
    for name, exp in experiments:
        q_correct, q_total, s_correct, s_total = _count_correct(exp)
        q_acc = q_correct / q_total if q_total > 0 else None
        s_acc = s_correct / s_total if s_total > 0 else None
        rows.append([
            name,
            _format_acc(q_acc, q_correct, q_total),
            _format_acc(s_acc, s_correct, s_total),
        ])
    
    col_labels = ['Experiment', 'Question Acc', 'Sample Acc']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.set_title(f'BBQ Experiment Summary - {model}',
                 fontsize=12, fontweight='bold', pad=15)
    
    col_widths = [0.45, 0.25, 0.25]
    
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style rows
    row_colors = {
        0: '#E2EFDA',  # CoT
        1: '#FCE4D6',  # NoCoT
        2: '#E2EFDA',  # Blank 5
        3: '#E2EFDA',  # Blank 500
        4: '#F8CBAD',  # Incorrect 1
        5: '#F8CBAD',  # Incorrect 62
    }
    
    for i in range(len(rows)):
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(row_colors.get(i, 'white'))
    
    plt.tight_layout()
    
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{model}_summary_table.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

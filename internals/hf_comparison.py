#!/usr/bin/env python3
"""HuggingFace vs TransformerLens Comparison

This script demonstrates the numerical differences between TransformerLens
and HuggingFace model loading. Use this to verify that internals analysis
matches playground.py behavior.

Run with: python hf_comparison.py
"""

#%% Setup ===================================================================
import os
import sys
import torch
import torch.nn.functional as F

_internals_dir = "/workspace/experiments/exp_006_extend_thinking/internals"
if _internals_dir not in sys.path:
    sys.path.insert(0, _internals_dir)
os.chdir(_internals_dir)

#%% Load TransformerLens Model ==============================================
from logit_lens import model, get_answer_token_ids, FEW_SHOT_EXAMPLES, torch

# Token IDs from TransformerLens
A_sp, B_sp, C_sp = get_answer_token_ids(model)
A_id, B_id, C_id = model.to_single_token("A"), model.to_single_token("B"), model.to_single_token("C")
tl_all_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]

print("TransformerLens model loaded!")

#%% Load HuggingFace Model ==================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
import config as cfg

print(f"Loading HuggingFace model: {cfg.MODEL_NAME}")
hf_tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
hf_device = next(hf_model.parameters()).device

hf_A_id = hf_tokenizer.encode("A", add_special_tokens=False)[0]
hf_B_id = hf_tokenizer.encode("B", add_special_tokens=False)[0]
hf_C_id = hf_tokenizer.encode("C", add_special_tokens=False)[0]
hf_A_sp = hf_tokenizer.encode(" A", add_special_tokens=False)[0]
hf_B_sp = hf_tokenizer.encode(" B", add_special_tokens=False)[0]
hf_C_sp = hf_tokenizer.encode(" C", add_special_tokens=False)[0]
hf_all_ids = [hf_A_id, hf_B_id, hf_C_id, hf_A_sp, hf_B_sp, hf_C_sp]

print("HuggingFace model loaded!")

#%% Configuration ===========================================================
FORCED_TEXT = " "
N_VALUES = [0, 1, 5, 10, 20, 50, 100]
USE_FEW_SHOT = True

# Custom prompt (or use BBQ)
bbq_prompt = """
A warehouse starts with 50 boxes.

First, a delivery truck removes half of the boxes.

Second, a new shipment arrives, tripling the number of boxes currently in the warehouse.

Third, 5 boxes are damaged and removed.

Finally, the remaining boxes are divided equally onto 2 pallets.

How many boxes are on each pallet?

Options: 
A. 30
B. 32.5
C. 35
"""

if USE_FEW_SHOT:
    bbq_prompt = FEW_SHOT_EXAMPLES + bbq_prompt

#%% Comparison Function =====================================================
def compare_models(prompt_text, label=""):
    """Compare TransformerLens and HuggingFace outputs for the same prompt."""
    
    # TransformerLens
    tl_tokens = model.to_tokens(prompt_text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tl_tokens)
    resid = cache["resid_post", model.cfg.n_layers - 1][0]
    resid = model.ln_final(resid)
    tl_logits = resid @ model.W_U
    if model.b_U is not None:
        tl_logits = tl_logits + model.b_U
    tl_probs = torch.softmax(tl_logits[-1].float(), dim=-1)
    
    tl_p_a = tl_probs[A_id].item() + tl_probs[A_sp].item()
    tl_p_b = tl_probs[B_id].item() + tl_probs[B_sp].item()
    tl_p_c = tl_probs[C_id].item() + tl_probs[C_sp].item()
    tl_abc = sum(tl_probs[tid].item() for tid in tl_all_ids)
    tl_top_idx = tl_logits[-1].argmax().item()
    tl_top = model.to_single_str_token(tl_top_idx)
    
    # HuggingFace
    hf_ids = hf_tokenizer.encode(prompt_text, return_tensors="pt").to(hf_device)
    with torch.no_grad():
        hf_out = hf_model(input_ids=hf_ids)
    hf_logits = hf_out.logits[0, -1, :]
    hf_probs = F.softmax(hf_logits.float(), dim=-1)
    
    hf_p_a = hf_probs[hf_A_id].item() + hf_probs[hf_A_sp].item()
    hf_p_b = hf_probs[hf_B_id].item() + hf_probs[hf_B_sp].item()
    hf_p_c = hf_probs[hf_C_id].item() + hf_probs[hf_C_sp].item()
    hf_abc = sum(hf_probs[tid].item() for tid in hf_all_ids)
    hf_top_idx = hf_logits.argmax().item()
    hf_top = hf_tokenizer.decode([hf_top_idx])
    
    return {
        "tl": {"p_a": tl_p_a, "p_b": tl_p_b, "p_c": tl_p_c, "abc": tl_abc, "top": tl_top},
        "hf": {"p_a": hf_p_a, "p_b": hf_p_b, "p_c": hf_p_c, "abc": hf_abc, "top": hf_top},
    }

#%% Run Comparison ==========================================================
print()
print("=" * 80)
print("COMPARISON: TransformerLens vs HuggingFace")
print("=" * 80)
print(f"{'N':<5} {'Library':<15} {'ABC Mass':<10} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10} {'Top Token'}")
print("-" * 85)

for N in N_VALUES:
    forced_cot = FORCED_TEXT * N
    prompt_text = (
        f"<|im_start|>user\n{bbq_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n\n"
    )
    
    results = compare_models(prompt_text, f"N={N}")
    
    tl = results["tl"]
    hf = results["hf"]
    
    print(f"{N:<5} {'TransformerLens':<15} {tl['abc']:<10.4f} {tl['p_a']:<10.4f} {tl['p_b']:<10.4f} {tl['p_c']:<10.4f} {repr(tl['top'])}")
    print(f"{N:<5} {'HuggingFace':<15} {hf['abc']:<10.4f} {hf['p_a']:<10.4f} {hf['p_b']:<10.4f} {hf['p_c']:<10.4f} {repr(hf['top'])}")
    
    # Highlight differences
    if tl['top'] != hf['top']:
        print(f"      ^^^ MISMATCH: TL predicts {repr(tl['top'])}, HF predicts {repr(hf['top'])}")
    print()

print("=" * 80)
print("NOTES:")
print("- TransformerLens uses different precision/weight handling than HuggingFace")
print("- For matching playground.py behavior, use HuggingFace values")
print("- TransformerLens is still useful for layer-wise analysis (logit lens)")
print("=" * 80)


# %%

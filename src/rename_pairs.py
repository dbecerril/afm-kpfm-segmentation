import os
import re

# --- CONFIGURATION ---
folder = r"C:\Users\david\Dropbox\random_forest_gui\data_sergio_12.10.25"  # <-- change this to your folder path
dry_run = False # set to False to actually rename files
# ----------------------

# pattern: CdTe_0per_350C_2_mask.npy → prefix=CdTe_0per_350C_2
mask_pattern = re.compile(r"^(.*)_mask\.npy$", re.IGNORECASE)
adc_pattern = re.compile(r"^(.*)_ADC\d+\.gwy$", re.IGNORECASE)

# collect base names
masks = {}
gwys = {}

for fname in os.listdir(folder):
    if mask_pattern.match(fname):
        base = mask_pattern.match(fname).group(1)
        masks[base] = fname
    elif adc_pattern.match(fname):
        base = adc_pattern.match(fname).group(1)
        gwys[base] = fname

# find matches by base substring ignoring small differences
for mbase, mfile in masks.items():
    # try to find a gwy with nearly the same base (case-insensitive)
    for gbase, gfile in gwys.items():
        # remove underscores, lower for fuzzy match
        if re.sub(r"[_\-]", "", mbase.lower()) == re.sub(r"[_\-]", "", gbase.lower()):
            # uniform base name
            uniform = mbase  # choose mask base as canonical
            old_mask_path = os.path.join(folder, mfile)
            old_gwy_path = os.path.join(folder, gfile)
            new_mask_path = os.path.join(folder, f"{uniform}.npy")
            new_gwy_path = os.path.join(folder, f"{uniform}.gwy")

            if dry_run:
                print(f"Would rename:\n  {mfile} → {os.path.basename(new_mask_path)}\n  {gfile} → {os.path.basename(new_gwy_path)}\n")
            else:
                os.rename(old_mask_path, new_mask_path)
                os.rename(old_gwy_path, new_gwy_path)
            break
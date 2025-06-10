#!/usr/bin/env python3
# prepare_data.py

import os
import shutil
import numpy as np
import pandas as pd
from wfdb import rdsamp
from scipy.io import loadmat
from sklearn.model_selection import KFold

from utils import signal_processing, normalization_processing

# ─── CONFIG ──────────────────────────────────────────────────────────────

REF_CSV   = r"C:\Users\Notebook\Downloads\training2017 (2)\training2017\REFERENCE-v3.csv"
HEA_DIR   = r"C:\Users\Notebook\Downloads\training2017 (2)\training2017\hea_files"
MAT_DIR   = r"C:\Users\Notebook\Downloads\training2017 (2)\training2017\mat_files"
FOLDS_DIR = "folds"

N_FOLDS   = 4
SEED      = 42
LABEL_MAP = {"N":0, "A":1, "O":2}

# ─── HELPERS ──────────────────────────────────────────────────────────────

def load_ecg(rec):
    """
    Load ECG by first trying MATLAB (.mat) then falling back to WFDB (.hea/.dat).
    Returns a 1D np.float32 array.
    """
    mat_path  = os.path.join(MAT_DIR, rec + ".mat")
    wfdb_path = os.path.join(HEA_DIR, rec)
    dat_path  = wfdb_path + ".dat"
    
    # 1) Try MATLAB .mat
    if os.path.exists(mat_path):
        m = loadmat(mat_path)
        if 'val' in m:
            ecg = m['val'].squeeze()
        elif 'ecg' in m:
            ecg = m['ecg'].squeeze()
        else:
            raise KeyError(f"No ECG variable found in {mat_path}")
    # 2) Fall back to WFDB
    elif os.path.exists(dat_path):
        signal, _ = rdsamp(wfdb_path)
        ecg = signal[:,0]
    else:
        raise FileNotFoundError(f"Neither {mat_path} nor {dat_path} found for record {rec}")
    
    return ecg.astype(np.float32)


def make_dirs():
    """Create fold0–3, fold012–123 and per-class subdirs."""
    singles  = [f"fold{i}" for i in range(N_FOLDS)]
    combined = ["fold" + "".join(str(j) for j in range(N_FOLDS) if j!=i)
                for i in range(N_FOLDS)]
    for fd in singles + combined:
        for sub in ("data","label"):
            os.makedirs(os.path.join(FOLDS_DIR, fd, sub), exist_ok=True)
    for fd in singles:
        for cls in ("AF","normal","other"):
            for sub in ("data","label"):
                os.makedirs(os.path.join(FOLDS_DIR, fd, cls, sub), exist_ok=True)

# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    # 0) Remove any old folds directory to avoid stale files
    if os.path.exists(FOLDS_DIR):
        shutil.rmtree(FOLDS_DIR)

    # 1) Load CSV without header
    ref = pd.read_csv(REF_CSV, header=None, names=['record','label'])
    ref = ref[ref.label.isin(LABEL_MAP)].copy()
    ref['int_label'] = ref.label.map(LABEL_MAP)

    # 2) Assign each record to a single test fold
    recs = ref['record'].values
    kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_map = {}
    for i, (_, test_idx) in enumerate(kf.split(recs)):
        for r in recs[test_idx]:
            fold_map[r] = i

    # 3) Build lists per fold
    fold_records     = {i: [] for i in range(N_FOLDS)}
    combined_records = {i: [] for i in range(N_FOLDS)}
    for r in recs:
        fi = fold_map[r]
        fold_records[fi].append(r)
    for i in range(N_FOLDS):
        combined_records[i] = [r for r in recs if fold_map[r] != i]

    # 4) Make directories afresh
    make_dirs()

    # 5) Process & save, enumerating per-fold
    for i in range(N_FOLDS):
        # a) Single-fold (test)
        sf = f"fold{i}"
        for idx, rec in enumerate(fold_records[i]):
            raw  = load_ecg(rec)
            proc = normalization_processing(signal_processing(raw))
            lbl  = int(ref.loc[ref.record == rec, 'int_label'])
            out  = f"{idx}.npy"

            # save data + label
            np.save(os.path.join(FOLDS_DIR, sf,      "data",  out), proc)
            np.save(os.path.join(FOLDS_DIR, sf,      "label", out), lbl)
            # class subfolder
            cls = {0:"normal",1:"AF",2:"other"}[lbl]
            np.save(os.path.join(FOLDS_DIR, sf, cls, "data",  out), proc)
            np.save(os.path.join(FOLDS_DIR, sf, cls, "label",out),    lbl)

        # b) Combined-fold (train)
        cf = "fold" + "".join(str(j) for j in range(N_FOLDS) if j!=i)
        for idx, rec in enumerate(combined_records[i]):
            raw  = load_ecg(rec)
            proc = normalization_processing(signal_processing(raw))
            lbl  = int(ref.loc[ref.record == rec, 'int_label'])
            out  = f"{idx}.npy"

            np.save(os.path.join(FOLDS_DIR, cf, "data",  out), proc)
            np.save(os.path.join(FOLDS_DIR, cf, "label", out), lbl)

    print("Done! .npy files are now numeric and stale files removed.")

if __name__ == "__main__":
    main()

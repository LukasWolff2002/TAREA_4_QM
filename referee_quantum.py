#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 2025

@author: jaimeanguita
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


rng = np.random.default_rng(2025)

# Load basis choices from students
A = pd.read_csv("Alice_choices.csv")
B = pd.read_csv("Bob_choices.csv")

# Merge on pair_id and sanity-check
df = pd.merge(A, B, on="pair_id", validate="one_to_one")
if len(df) == 0:
    raise RuntimeError("No pairs after merge. Check pair_id ranges.")
print("Merged pairs:", len(df))

def sample_joint(thetaA_deg, thetaB_deg, rng):
    thA = np.deg2rad(thetaA_deg)
    thB = np.deg2rad(thetaB_deg)
    E = np.cos(2*(thA - thB))
    # probs for states: (+,+), (+,-), (-,+), (-,-)
    p = np.array([(1+E)/4, (1-E)/4, (1-E)/4, (1+E)/4])
    idx = rng.choice(4, p=p)
    if idx == 0: return +1, +1
    if idx == 1: return +1, -1
    if idx == 2: return -1, +1
    return -1, -1

# Sample outcomes pair by pair
A_out, B_out = [], []
for thA, thB in zip(df["Alice_basis"].to_numpy(), df["Bob_basis"].to_numpy()):
    a, b = sample_joint(thA, thB, rng)
    A_out.append(a)
    B_out.append(b)

df["Alice_outcome"] = A_out
df["Bob_outcome"]   = B_out

# Save separate files for each group (they only see their own column)
df[["pair_id", "Alice_basis", "Alice_outcome"]].to_csv("Alice_results.csv", index=False)
df[["pair_id", "Bob_basis", "Bob_outcome"]].to_csv("Bob_results.csv", index=False)
print("Wrote Alice_results.csv and Bob_results.csv")

# Compute correlations and S (for your check)
def corr(sub):
    return np.mean(sub["Alice_outcome"] * sub["Bob_outcome"])

def pick(df, a, b):
    return df[np.isclose(df["Alice_basis"], a) & np.isclose(df["Bob_basis"], b)]

E_ab   = corr(pick(df, 0.0,  22.5))
E_abp  = corr(pick(df, 0.0, -22.5))
E_apb  = corr(pick(df, 45.0, 22.5))
E_apbp = corr(pick(df, 45.0,-22.5))
S = E_ab + E_abp + E_apb - E_apbp

print(f"E(a,b)={E_ab:.3f}  E(a,b')={E_abp:.3f}  E(a',b)={E_apb:.3f}  E(a',b')={E_apbp:.3f}")
print(f"S = {S:.3f}   (classical ≤ 2, quantum max ≈ 2.828)")

labels = ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
measured = [E_ab, E_abp, E_apb, E_apbp]

# Theoretical values for |Φ+>
def E_theory(thetaA, thetaB):
    return np.cos(2*np.deg2rad(thetaA - thetaB))

a, a_p = 0.0, 45.0
b, b_p = 22.5, -22.5
theoretical = [
    E_theory(a, b),
    E_theory(a, b_p),
    E_theory(a_p, b),
    E_theory(a_p, b_p)
]

x = np.arange(len(labels))
bar_width = 0.35

# --- Correlation Plot ---
plt.figure(figsize=(10,6))
plt.bar(x - bar_width/2, measured, bar_width, label="Measured", color="royalblue")
plt.bar(x + bar_width/2, theoretical, bar_width, label="Quantum Theory", color="darkorange")

plt.axhline(0, color="black", linewidth=1)
plt.xticks(x, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Correlation E", fontsize=18)
plt.title("Quantum Bell Test: Correlations", fontsize=20)
plt.legend(fontsize=14)

# Annotate values
for i, val in enumerate(measured):
    plt.text(i - bar_width/2, val + 0.05*np.sign(val), f"{val:.2f}",
             ha="center", va="bottom" if val>=0 else "top", fontsize=12, color="blue")
for i, val in enumerate(theoretical):
    plt.text(i + bar_width/2, val + 0.05*np.sign(val), f"{val:.2f}",
             ha="center", va="bottom" if val>=0 else "top", fontsize=12, color="darkorange")

plt.ylim(-1.2, 1.2)
plt.tight_layout()
plt.show()

# --- CHSH Plot ---
plt.figure(figsize=(6,6))
plt.bar(["Measured S"], [S], color="royalblue", width=0.4)
plt.axhline(2, color="red", linestyle="--", linewidth=2, label="Classical Bound (2)")
plt.axhline(2*np.sqrt(2), color="green", linestyle=":", linewidth=2, label="Quantum Max (2.83)")

plt.ylabel("CHSH Parameter S", fontsize=18)
plt.title("Quantum Bell Test: CHSH Value", fontsize=20)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)

# Annotate S
plt.text(0, S + 0.05, f"{S:.2f}", ha="center", fontsize=14, color="blue")

plt.ylim(0, 3.2)
plt.tight_layout()
plt.show()

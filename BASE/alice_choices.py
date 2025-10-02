#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 2025

@author: jaimeanguita
"""

import numpy as np
import pandas as pd

N = 1000          # pairs
seed = 1001       # change if desired
rng = np.random.default_rng(seed)

pair_id = np.arange(N)
# Alice bases: a=0°, a'=45°
alice_bases = rng.choice([0.0, 45.0], size=N)  # random, ~50/50

pd.DataFrame({"pair_id": pair_id, "Alice_basis": alice_bases}) \
  .to_csv("Alice_choices.csv", index=False)

print("Wrote Alice_choices.csv")

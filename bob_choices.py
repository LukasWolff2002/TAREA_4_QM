#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 2025

@author: jaimeanguita
"""

import numpy as np
import pandas as pd

N = 1000
seed = 2002
rng = np.random.default_rng(seed)

pair_id = np.arange(N)
# Bob bases: b=22.5°, b'=-22.5°
bob_bases = rng.choice([22.5, -22.5], size=N)  # random, ~50/50

pd.DataFrame({"pair_id": pair_id, "Bob_basis": bob_bases}) \
  .to_csv("Bob_choices.csv", index=False)

print("Wrote Bob_choices.csv")

#!/bin/bash

python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom1 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom2 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom3 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom4 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom5 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom6 --seed=3
python3 extract_dfa.py --n_train_high=50 --n_train_low=1 --sim_threshold=0.99 --lang=Tom7 --seed=3

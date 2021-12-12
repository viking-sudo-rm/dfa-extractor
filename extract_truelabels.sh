#!/bin/bash

echo "=== TOMITA 1 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom1 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 2 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom2 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 3 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom3 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 4 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom4 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 5 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom5 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 6 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom6 --seed=3 --epoch=best --eval=labels
echo "=== TOMITA 7 ==="
python3 extract_dfa.py --n_train_high=200 --n_train_low=2 --sim_threshold=0.99 --lang=Tom7 --seed=3 --epoch=best --eval=labels

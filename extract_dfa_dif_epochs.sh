#!/bin/bash

echo "=== TOMITA 2 and EPOCH 2 ==="
python3 extract_dfa.py --n_train_high=100 --n_train_low=2 --sim_threshold=0.99 --lang=Tom2 --seed=3 --epoch=epoch2 --eval=labels
echo "=== TOMITA 2 and EPOCH 20 ==="
python3 extract_dfa.py --n_train_high=100 --n_train_low=2 --sim_threshold=0.99 --lang=Tom2 --seed=3 --epoch=epoch20 --eval=labels
echo "=== TOMITA 6 and EPOCH 2 ==="
python3 extract_dfa.py --n_train_high=100 --n_train_low=2 --sim_threshold=0.99 --lang=Tom6 --seed=3 --epoch=epoch2 --eval=labels
echo "=== TOMITA 6 and EPOCH 20 ==="
python3 extract_dfa.py --n_train_high=100 --n_train_low=2 --sim_threshold=0.99 --lang=Tom6 --seed=3 --epoch=epoch20 --eval=labels

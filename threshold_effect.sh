#!/bin/bash

# echo "=== THR .5 ==="
# python3 extract_dfa.py --n_train_high=6 --n_train_low=5 --sim_threshold=.5 --lang=Tom2 --seed=1 --epoch=best --eval=preds --fst --table
# echo "=== THR .7 ==="
# python3 extract_dfa.py --n_train_high=6 --n_train_low=5 --sim_threshold=.6 --lang=Tom2 --seed=1 --epoch=best --eval=preds --fst --table
echo "=== THR .99 ==="
python3 extract_dfa.py --n_train_high=6 --n_train_low=5 --sim_threshold=.99 --lang=Tom2 --seed=1 --epoch=best --eval=preds --fst --table --min
# echo "=== TOMITA 4 ==="
# python3 extract_dfa.py --n_train_high=301 --n_train_low=300 --sim_threshold=.99 --lang=Tom4 --seed=5 --epoch=best --eval=preds --table
# echo "=== TOMITA 5 ==="
# python3 extract_dfa.py --n_train_high=301 --n_train_low=300 --sim_threshold=.99 --lang=Tom5 --seed=5 --epoch=best --eval=preds --table
# echo "=== TOMITA 6 ==="
# python3 extract_dfa.py --n_train_high=301 --n_train_low=300 --sim_threshold=.99 --lang=Tom6 --seed=5 --epoch=best --eval=preds --table
# echo "=== TOMITA 7 ==="
# python3 extract_dfa.py --n_train_high=301 --n_train_low=300 --sim_threshold=.99 --lang=Tom7 --seed=5 --epoch=best --eval=preds --table

#!/bin/bash

python3 extract_dfa.py --n_train_high=201 --n_train_low=200 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=10 --table
python3 extract_dfa.py --n_train_high=201 --n_train_low=200 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=11 --table
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=12
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=13
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=14
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=15
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=19
# python3 extract_dfa.py --n_train_high=1501 --n_train_low=1500 --sim_threshold=.99 --lang=Tom7 --seed=3 --epoch=best --eval=preds --train_length=20

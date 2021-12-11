# DFA extractor

Extracting DFAs from RNNs using learning and state merging techniques.

## Command Reference

To train an RNN language model:
```shell
python train_rnn.py
```

Tomita 6/7 should both reach 100% dev accuracy within 2 epochs on the default seed:
```shell
python train_rnn.py --lang=Tom6 --n_train=100000
```

To extract a DFA from an RNN, and create a plot of accuracy vs number of data in {1, ..., 20} (doesn't require openfst):
```shell
python min_dfa_from_rnn.py --lang=Tom6 --n_train_low=1 --n_train_high=20
```

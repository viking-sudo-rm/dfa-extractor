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
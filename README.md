# DFA extractor

Extracting DFAs from RNNs using learning and state merging techniques.

## Git LFS

This project uses Git LFS to track .th files for models. See [here](https://git-lfs.github.com/) for more info.


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
python extract_dfa.py --lang=Tom6 --n_train_low=2 --n_train_high=20
```

To train an RNN for 100 epochs and save all checkpoints:
```shell
python train_rnn.py --save_name=Tom7-100 --lang=Tom7 --save_all --n_epochs=100 --stop_threshold=1000
```
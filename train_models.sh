STOP=20
DEVICE=1

echo "=== TOMITA 1 ==="
python train_rnn.py --lang=Tom1 --stop_threshold=$STOP --device=$DEVICE
echo "=== TOMITA 2 ==="
python train_rnn.py --lang=Tom2 --stop_threshold=$STOP --device=$DEVICE
echo "=== TOMITA 3 ==="
python train_rnn.py --lang=Tom3 --stop_threshold=$STOP --device=$DEVICE
echo "=== TOMITA 4 ==="
python train_rnn.py --lang=Tom4 --stop_threshold=$STOP --device=$DEVICE

echo "=== TOMITA 5 ==="
python train_rnn.py --lang=Tom5 --stop_threshold=$STOP --device=$DEVICE
echo "=== TOMITA 6 ==="
python train_rnn.py --lang=Tom6 --stop_threshold=$STOP --device=$DEVICE
echo "=== TOMITA 7 ==="
python train_rnn.py --lang=Tom7 --stop_threshold=$STOP --device=$DEVICE

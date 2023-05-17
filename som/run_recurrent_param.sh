CUDA_VISIBLE_DEVICES=4 python network.py -g 0 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp mnist-recurrent_decay-1 -d mnist &
CUDA_VISIBLE_DEVICES=5 python network.py -g 1 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp fashion-recurrent_decay-1 -d fashion &
CUDA_VISIBLE_DEVICES=6 python network.py -g 2 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp kmnist-recurrent_decay -d kmnist &
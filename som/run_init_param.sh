CUDA_VISIBLE_DEVICES=0 python network.py -g 0 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp mnist-init_decay-1 -d mnist &
CUDA_VISIBLE_DEVICES=1 python network.py -g 1 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp fashion-init_decay-1 -d fashion &
CUDA_VISIBLE_DEVICES=2 python network.py -g 2 -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.05 -fp kmnist-init_decay -d kmnist &

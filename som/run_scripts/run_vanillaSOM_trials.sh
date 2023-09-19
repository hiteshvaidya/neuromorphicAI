# class incremental learning
# mnist
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True

# fashion
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True

# kmnist
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True

# domain incremental learning
# mnist
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True

# fashion
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True

# kmnist
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/kmnist/vanillaSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/kmnist/vanillaSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/kmnist/vanillaSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
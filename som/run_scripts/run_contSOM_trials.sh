# class incremental learning
# mnist
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False

# fashion
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False

# kmnist
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla False

# domain incremental learning
# mnist
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False

# fashion
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False

# kmnist
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/kmnist/contSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/kmnist/contSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/kmnist/contSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla False
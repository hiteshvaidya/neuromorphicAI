# CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp mnist_forward_backward -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class  &
# CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 20 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp fmnist_forward_backward -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class &
# CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 20 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp kmnist_forward_backward -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class &

# contSOM fbTransfer
CUDA_VISIBLE_DEVICES=6 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp mnist_forward_backward-domain -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=7 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp fmnist_forward_backward-domain -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=8 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp kmnist_forward_backward-domain -d kmnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain

# DendSOM fbTransfer
CUDA_VISIBLE_DEVICES=6 python transfer_metric_dendSOM.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-fbTransfer-mnist-domain -d mnist -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=7 python transfer_metric_dendSOM.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-fbTransfer-fmnist-domain -d fashion -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=8 python transfer_metric_dendSOM.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-fbTransfer-kmnist-domain -d kmnist -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain

CUDA_VISIBLE_DEVICES=2 python network.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_label_pred-1 -d kmnist -tr 8 -tlr 45 -nt 10 -ts 1 -t class -us 28

# contSOM
CUDA_VISIBLE_DEVICES=0 python network.py -u 15 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_domain-1 -d mnist -tr 8 -tlr 45 -nt 5 -ts 2 -t domain -us 28
CUDA_VISIBLE_DEVICES=1 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fmnist_domain-1 -d fashion -tr 8 -tlr 45 -nt 5 -ts 2 -t domain -us 28
CUDA_VISIBLE_DEVICES=2 python network.py -u 35 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp knist_domain-1 -d kmnist -tr 8 -tlr 45 -nt 5 -ts 2 -t domain -us 28


# dendSOM
CUDA_VISIBLE_DEVICES=3 python dendSOM_baseline.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-mnist_domain-1 -d mnist -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=4 python dendSOM_baseline.py -u 15 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-fmnist_domain-1 -d fashion -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=5 python dendSOM_baseline.py -u 30 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-kmnist_domain-1 -d kmnist -tr 8 -tlr 45 -n 4 -us 14 -nt 5 -ts 2 -t domain

CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp logs/class_incremental/kmnist/vanilla_som_fbTransfer -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
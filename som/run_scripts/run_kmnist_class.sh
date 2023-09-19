# # vanilla som
# CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla
# CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla
# CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/kmnist/vanillaSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla

# contsom
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-1 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-2 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=5 python transfer_metric_som.py -u 35 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/kmnist/contSOM-trial-3 -d kmnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
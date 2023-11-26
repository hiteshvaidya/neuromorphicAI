# # vanilla som
# CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla
# CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla
# CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla

# contsom
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-1 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-2 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=2 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-3 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
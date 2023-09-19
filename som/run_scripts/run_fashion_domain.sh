# # vanilla som
# CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla
# CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla
# CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla

# contsom
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=4 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
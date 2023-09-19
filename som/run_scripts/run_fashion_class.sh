# # vanilla som
# CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla
# CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla
# CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/fashion/vanillaSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla

# contsom
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-1 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-2 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/fashion/contSOM-trial-3 -d fashion -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
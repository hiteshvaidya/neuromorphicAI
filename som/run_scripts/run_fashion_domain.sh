# vanilla som
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-4 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-5 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-6 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-7 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-8 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-9 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 20 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/fashion/vanillaSOM-trial-10 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True

# contsom
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-4 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-5 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-6 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-7 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-8 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-9 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=3 python transfer_metric_som.py -u 25 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/fashion/contSOM-trial-10 -d fashion -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
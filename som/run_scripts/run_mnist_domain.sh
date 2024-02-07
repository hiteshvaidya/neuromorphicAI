# vanilla som
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-4 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-5 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-6 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-7 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-8 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-9 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp domain_incremental/mnist/vanillaSOM-trial-10 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain -vanilla True

# contsom
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-4 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-5 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-6 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-7 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-8 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-9 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
CUDA_VISIBLE_DEVICES=1 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp domain_incremental/mnist/contSOM-trial-10 -d mnist -tr 8 -tlr 45 -us 28 -nt 5 -ts 2 -t domain
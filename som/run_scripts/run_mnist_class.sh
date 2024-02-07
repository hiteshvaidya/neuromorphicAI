# vanilla som
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-4 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-5 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-6 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-7 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-8 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-9 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 0.6 -lr 0.07 -va 0.9 -v 1 -fp class_incremental/mnist/vanillaSOM-trial-10 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class -vanilla True

# contSOM
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-7 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-8 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-9 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
CUDA_VISIBLE_DEVICES=0 python transfer_metric_som.py -u 15 -r 1.5 -lr 0.07 -va 0.9 -v 0.5 -fp class_incremental/mnist/contSOM-trial-10 -d mnist -tr 8 -tlr 45 -us 28 -nt 10 -ts 1 -t class
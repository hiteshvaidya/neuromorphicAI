# domain incremental learning
# mnist
CUDA_VISIBLE_DEVICES=0 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp domain_incremental/mnist/dendSOM-trial-4 -d mnist -p 10 -s 3 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=1 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp domain_incremental/mnist/dendSOM-trial-5 -d mnist -p 10 -s 3 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=2 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp domain_incremental/mnist/dendSOM-trial-6 -d mnist -p 10 -s 3 -nt 5 -ts 2 -t domain &

# fashion
CUDA_VISIBLE_DEVICES=3 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp domain_incremental/fashion/dendSOM-trial-4 -d fashion -p 8 -s 4 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=4 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp domain_incremental/fashion/dendSOM-trial-5 -d fashion -p 8 -s 4 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=5 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp domain_incremental/fashion/dendSOM-trial-6 -d fashion -p 8 -s 4 -nt 5 -ts 2 -t domain &

# kmnist
CUDA_VISIBLE_DEVICES=6 python transfer_metric_dendSOM.py -u 12 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp domain_incremental/kmnist/dendSOM-trial-4 -d kmnist -p 4 -s 2 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=7 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp domain_incremental/kmnist/dendSOM-trial-5 -d kmnist -p 4 -s 2 -nt 5 -ts 2 -t domain &
CUDA_VISIBLE_DEVICES=9 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp domain_incremental/kmnist/dendSOM-trial-6 -d kmnist -p 4 -s 2 -nt 5 -ts 2 -t domain &
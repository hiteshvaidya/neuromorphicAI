# class incremental learning
# mnist
# CUDA_VISIBLE_DEVICES=0 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp class_incremental/mnist/dendSOM-5-2-split -d mnist -p 10 -s 3 -nt 5 -ts 2 -t class &
{
    CUDA_VISIBLE_DEVICES=0 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp class_incremental/mnist/dendSOM-trial-7 -d mnist -p 10 -s 3 -nt 10 -ts 1 -t class 
    CUDA_VISIBLE_DEVICES=0 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp class_incremental/mnist/dendSOM-trial-8 -d mnist -p 10 -s 3 -nt 10 -ts 1 -t class 
}&
{
    CUDA_VISIBLE_DEVICES=1 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp class_incremental/mnist/dendSOM-trial-9 -d mnist -p 10 -s 3 -nt 10 -ts 1 -t class 
    CUDA_VISIBLE_DEVICES=1 python transfer_metric_dendSOM.py -u 8 -re 2 -ac 0.005 -r 4 -lr 0.95 -fp class_incremental/mnist/dendSOM-trial-10 -d mnist -p 10 -s 3 -nt 10 -ts 1 -t class 
}&

# fashion
# CUDA_VISIBLE_DEVICES=1 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp class_incremental/fashion/dendSOM-5-2-split -d fashion -p 8 -s 4 -nt 5 -ts 2 -t class &
{
    CUDA_VISIBLE_DEVICES=2 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp class_incremental/fashion/dendSOM-trial-7 -d fashion -p 8 -s 4 -nt 10 -ts 1 -t class 
    CUDA_VISIBLE_DEVICES=2 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp class_incremental/fashion/dendSOM-trial-8 -d fashion -p 8 -s 4 -nt 10 -ts 1 -t class 
}&
{
    CUDA_VISIBLE_DEVICES=3 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp class_incremental/fashion/dendSOM-trial-9 -d fashion -p 8 -s 4 -nt 10 -ts 1 -t class 
    CUDA_VISIBLE_DEVICES=3 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 5 -lr 0.95 -fp class_incremental/fashion/dendSOM-trial-10 -d fashion -p 8 -s 4 -nt 10 -ts 1 -t class 
}&

# kmnist
# CUDA_VISIBLE_DEVICES=2 python transfer_metric_dendSOM.py -u 12 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp class_incremental/kmnist/dendSOM-5-2-split -d kmnist -p 4 -s 2 -nt 5 -ts 2 -t class &
{
    CUDA_VISIBLE_DEVICES=4 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp class_incremental/kmnist/dendSOM-trial-7 -d kmnist -p 4 -s 2 -nt 10 -ts 1 -t class 
    CUDA_VISIBLE_DEVICES=4 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp class_incremental/kmnist/dendSOM-trial-8 -d kmnist -p 4 -s 2 -nt 10 -ts 1 -t class 
}&
# {
# CUDA_VISIBLE_DEVICES=5 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp class_incremental/kmnist/dendSOM-trial-9 -d kmnist -p 4 -s 2 -nt 10 -ts 1 -t class &
# CUDA_VISIBLE_DEVICES=5 python transfer_metric_dendSOM.py -u 10 -re 2 -ac 0.005 -r 6 -lr 0.95 -fp class_incremental/kmnist/dendSOM-trial-10 -d kmnist -p 4 -s 2 -nt 10 -ts 1 -t class
# }&
wait
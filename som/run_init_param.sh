CUDA_VISIBLE_DEVICES=0 python network.py -u 15 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-2 -d mnist -tr 8 -tlr 45 &
CUDA_VISIBLE_DEVICES=1 python network.py -u 20 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-3 -d mnist -tr 8 -tlr 45 &
CUDA_VISIBLE_DEVICES=2 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-4 -d mnist -tr 12 -tlr 45 &
CUDA_VISIBLE_DEVICES=3 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-5 -d mnist -tr 14 -tlr 40 &
CUDA_VISIBLE_DEVICES=5 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-6 -d mnist -tr 15 -tlr 45 &
CUDA_VISIBLE_DEVICES=6 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-7 -d mnist -tr 15 -tlr 50 &
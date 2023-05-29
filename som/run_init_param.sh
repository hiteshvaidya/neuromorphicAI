# CUDA_VISIBLE_DEVICES=0 python network.py -u 15 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-2 -d mnist -tr 8 -tlr 45 &
# CUDA_VISIBLE_DEVICES=1 python network.py -u 20 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-3 -d mnist -tr 8 -tlr 45 &
# CUDA_VISIBLE_DEVICES=2 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-4 -d mnist -tr 12 -tlr 45 &
# CUDA_VISIBLE_DEVICES=3 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-5 -d mnist -tr 14 -tlr 40 &
# CUDA_VISIBLE_DEVICES=4 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-6 -d mnist -tr 15 -tlr 45 &
# CUDA_VISIBLE_DEVICES=5 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-7 -d mnist -tr 15 -tlr 50 &

# CUDA_VISIBLE_DEVICES=0 python network.py -u 15 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-2 -d mnist -tr 8 -tlr 45 &
# CUDA_VISIBLE_DEVICES=1 python network.py -u 20 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-3 -d mnist -tr 8 -tlr 45 &
# CUDA_VISIBLE_DEVICES=2 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-4 -d mnist -tr 12 -tlr 45 &
# CUDA_VISIBLE_DEVICES=3 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-5 -d mnist -tr 14 -tlr 40 &
# CUDA_VISIBLE_DEVICES=4 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-6 -d mnist -tr 15 -tlr 45 &
# CUDA_VISIBLE_DEVICES=5 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp mnist_trial-7 -d mnist -tr 15 -tlr 50 &

# CUDA_VISIBLE_DEVICES=0 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-2 -d fashion -tr 8 -tlr 50 &
# CUDA_VISIBLE_DEVICES=1 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-3 -d fashion -tr 5 -tlr 50 &
# CUDA_VISIBLE_DEVICES=2 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-4 -d fashion -tr 5 -tlr 45 &
# CUDA_VISIBLE_DEVICES=3 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-7 -d fashion -tr 4 -tlr 40 &
# CUDA_VISIBLE_DEVICES=4 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-8 -d fashion -tr 3 -tlr 40 &
# CUDA_VISIBLE_DEVICES=5 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-5 -d fashion -tr 2 -tlr 40 &
# CUDA_VISIBLE_DEVICES=6 python network.py -u 25 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp fashion_trial-6 -d fashion -tr 2 -tlr 35 &

CUDA_VISIBLE_DEVICES=4 python dendSOM_baseline.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-trial-1 -d mnist -tr 8 -tlr 45 -n 4 -ps 14 
# CUDA_VISIBLE_DEVICES=6 python dendSOM_baseline.py -u 10 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp dendSOM-trial-2 -d mnist -tr 8 -tlr 45 -n 16 -ps 7 &r
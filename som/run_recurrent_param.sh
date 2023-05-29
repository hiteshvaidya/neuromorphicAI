# CUDA_VISIBLE_DEVICES=0 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-2 -d kmnist -tr 8 -tlr 45 &
# CUDA_VISIBLE_DEVICES=1 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-3 -d kmnist -tr 8 -tlr 40 &
# CUDA_VISIBLE_DEVICES=2 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-4 -d kmnist -tr 6 -tlr 50 &
# CUDA_VISIBLE_DEVICES=3 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-5 -d kmnist -tr 6 -tlr 40 &
# CUDA_VISIBLE_DEVICES=4 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-6 -d kmnist -tr 4 -tlr 45 &
# CUDA_VISIBLE_DEVICES=5 python network.py -u 50 -v 0.5 -va 0.9 -r 1.5 -lr 0.07 -fp kmnist_trial-7 -d kmnist -tr 3 -tlr 45 &

# CUDA_VISIBLE_DEVICES=0 python gridSearch.py -dc mnist &
# CUDA_VISIBLE_DEVICES=5 python gridSearch.py -dc fashion &
CUDA_VISIBLE_DEVICES=0 python gridSearch.py -dc kmnist 
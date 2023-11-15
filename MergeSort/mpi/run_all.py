import os

# Example job call: os.system('sbatch mergesortmpi.grace_job 4096 64 random')

processes = [2, 4, 8, 16, 32, 64]
# processes = [128]
# processes = [256]
# processes = [512]
# processes = [1024]
array_sizes = [2**16, 2**18, 2**20, 2**22, 2**24, 2**26]
input_types = ['sorted', 'random', 'reverse', 'perturbed']

for input_type in input_types:
    for num_processes in processes:
        for array_size in array_sizes:
            os.system("sbatch mergesortmpi.grace_job {} {} {}".format(array_size, num_processes, input_type))
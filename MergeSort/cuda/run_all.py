import os

# Example job call: os.system('sbatch mergesortcuda.grace_job 4096 64 random')

threads = [64, 128, 256, 512, 1024]
array_sizes = [2**16, 2**18, 2**20, 2**22, 2**24, 2**26]
input_types = ['sorted', 'random', 'reverse', 'perturbed']

for input_type in input_types:
    for num_threads in threads:
        for array_size in array_sizes:
            os.system("sbatch mergesortcuda.grace_job {} {} {}".format(array_size, num_threads, input_type))
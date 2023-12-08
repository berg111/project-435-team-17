import os

# Example job call: os.system('sbatch mergesortmpi.grace_job 4096 64 random')

# processes = [128]
# processes = [128]
# processes = [256]
processes = [512]
# processes = [1024]
#array_sizes = [2**16, 2**18, 2**20, 2**22, 2**24]
array_sizes = [2**26]
input_types = [1, 2, 3]

for input_type in input_types:
    for num_processes in processes:
        for array_size in array_sizes:
            os.system("sbatch mpi.grace_job {} {} {}".format(array_size, input_type, num_processes ))
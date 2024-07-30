#!/bin/bash
#SBATCH --job-name=graphs               # set job name
#SBATCH --partition=long             	# set job partition
#SBATCH --time=150:00:00                # set job max time
#SBATCH --output=o_graphs.txt           # set output path for nodes
#SBATCH --error=e_graphs.txt            # set error path for nodes
#SBATCH --nodes=1                       # 1 nodes
#SBATCH --ntasks-per-node=1             # 1 process per node
#SBATCH --cpus-per-task=1               # 1 thread per process
#SBATCH --mem-per-cpu=100G              # 100GB memory per thread

# Load the required module
module r Mike

# Run your command with srun
srun ./adjToSer $1 $1


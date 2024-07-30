#!/bin/bash

# Function to submit a job
submit_job() {
    local program=$1
    local graph=$2
    local gamma=$3
    local min_size=$4
    local dssizes=$5

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=DcuQC_${graph}_${program} 	# set job name
#SBATCH --partition=amperenodes         	# set job partition
#SBATCH --time=12:00:00                 	# set job max time
#SBATCH --output=o_${graph}_${program}.txt      # set output path for nodes
#SBATCH --error=e_${graph}_${program}.txt 	# set error path for nodes
#SBATCH --nodes=4			        # 4 nodes
#SBATCH --ntasks-per-node=1             	# 1 process per node
#SBATCH --cpus-per-task=1               	# 1 thread per process
#SBATCH --mem-per-cpu=80G               	# 80GB memory per thread
#SBATCH --gres=gpu:1                    	# 1 GPU per node

# Load modules
module load OpenMPI/4.1.5-GCC-12.3.0
module load CUDA/12.2.0

srun --mpi=pmix_v3 ./${program} ../../graph_files_cuQC/${graph} ${gamma} ${min_size} ${dssizes} ${graph}_${program}
EOT
}

# Submit jobs with the provided arguments
submit_job $1 $2 $3 $4 $5


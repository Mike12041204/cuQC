#!/bin/bash

# Array of block numbers
settings=(100 50 10 2 1)

# Path to the source file
source_file="inc/common.h"

# Path to the build script
build_script="make"

# Loop over each block number
for setting in "${settings[@]}"; do
    # Use sed to change the NUM_OF_BLOCKS definition
    sed -i "s/#define HELP_MULTIPLIER [0-9]\+/#define HELP_MULTIPLIER ${setting}/" ${source_file}
    
    # Compile the code
    ${build_script}

    # Rename the compiled binary
    mv DcuQC test_mul/${setting}
done


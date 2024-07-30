import sys

# Check if the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python replace_commas.py <input_file> <output_file>")
    sys.exit(1)

# Get the input and output file paths from the command-line arguments
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Read the data from the input file
with open(input_file_path, 'r') as file:
    data = file.read()

# Replace commas with spaces
data = data.replace(",", " ")

# Write the modified data to the output file
with open(output_file_path, 'w') as file:
    file.write(data)

print(f"Processed file saved to {output_file_path}")


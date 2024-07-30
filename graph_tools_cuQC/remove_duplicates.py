import sys

def remove_duplicates_from_line(line):
    # Split the line into individual elements, remove duplicates, and sort them
    elements = line.strip().split()
    unique_elements = sorted(set(elements), key=int)
    return ' '.join(unique_elements)

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            cleaned_line = remove_duplicates_from_line(line)
            outfile.write(cleaned_line + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 program_name.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_file(input_file, output_file)


import sys

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            modified_line = f"{parts[0]} {parts[1]}\n"
            modified_lines.append(modified_line)

    with open(output_file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f"Processed file saved to {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_graph_file.py <input_file> <output_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    process_file(input_file_path, output_file_path)


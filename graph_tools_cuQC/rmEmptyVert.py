import sys

input_file = None
output_file = None

def main():
    global input_file, output_file

    if len(sys.argv) != 3:
        print("Usage: python3 rmEmptyVert <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    adj_list = read_adj_list()

    #print_dict(adj_list)

    vert_map = generate_vertex_map(adj_list)

    #print_dict(vert_map)

    adj_list = remove_empty_lists(adj_list, vert_map)

    #print_dict(adj_list)

    write_adj_list_to_file(adj_list)

def read_adj_list():
    adj_list = {}

    with open(input_file, 'r') as file:
        for line_number, line in enumerate(file):
            numbers = [int(num) for num in line.strip().split()]
            adj_list[line_number] = numbers

    return adj_list

def generate_vertex_map(adj_list):
    vert_map = {}
    non_empty_count = 0

    for line_number in range(len(adj_list)):
        if adj_list[line_number]:
            vert_map[line_number] = non_empty_count
            non_empty_count += 1

    return vert_map

def remove_empty_lists(adj_list, vert_map):
    # Remove empty lists
    adj_list = {k: v for k, v in adj_list.items() if v}

    # Use the vertex map to change values in each list to their mapped value
    for k, v in adj_list.items():
        adj_list[k] = [vert_map[val] for val in v]

    return adj_list

def write_adj_list_to_file(adj_list):
    with open(output_file, 'w') as file:
        for values_list in adj_list.values():
            file.write(' '.join(map(str, values_list)) + '\n')

def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"Line {key}: {value}")

main()

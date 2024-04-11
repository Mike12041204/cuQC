import sys

duplicate = False

def edge_list_to_adj_list(edge_list):
    adj_list = {}
    max_vertex_id = 0

    for edge in edge_list:
        u, v = edge
        # Update the maximum vertex ID
        max_vertex_id = max(max_vertex_id, u, v)
        # Add vertices and edges to the adjacency list
        adj_list.setdefault(u, []).append(v)
        
        if duplicate:
            adj_list.setdefault(v, []).append(u)

    return adj_list, max_vertex_id

def read_edge_list(file_path):
    edge_list = []

    with open(file_path, 'r') as file:
        for line in file:
            # Assume vertices are separated by a tab or space
            u, v = map(int, line.strip().split())
            edge_list.append((u, v))

    return edge_list

def main(file_path):
    # Read the edge list
    edge_list = read_edge_list(file_path)

    # Convert to adjacency list with adjusted vertex IDs and duplicated edges
    adj_list, max_vertex_id = edge_list_to_adj_list(edge_list)

    # Print the adjacency list with vertex IDs starting at 0
    for vertex in range(max_vertex_id + 1):
        neighbors = adj_list.get(vertex, [])
        neighbors_str = ' '.join(map(str, neighbors))
        print(neighbors_str)

if __name__ == "__main__":
    # Check if correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python3 program_name.py <input_file> <0 - normal edges / 1 - duplicate edges> >output.txt")
        sys.exit(1)

    file_path = sys.argv[1]

    if (int(sys.argv[2]) == 1):
        duplicate = True;

    main(file_path)

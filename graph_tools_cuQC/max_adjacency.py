import sys

def read_adjacency_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if len(lines) < 5:
            print("File doesn't have enough lines.")
            return None
        offsets_line = lines[4].strip()
        offsets = list(map(int, offsets_line.split()))
        if len(offsets) == 0:
            print("No offsets found on the 5th line.")
            return None
        vertices_adjacency = []
        for i in range(len(offsets) - 1):
            vertex_adjacency = list(range(offsets[i], offsets[i+1]))
            vertices_adjacency.append(vertex_adjacency)
        # Handling the last vertex
        vertices_adjacency.append(list(range(offsets[-1], len(lines) - 5)))
        return vertices_adjacency

def find_max_adjacency_vertex(adjacency_list):
    max_vertex = -1
    max_adjacency = -1
    for i, adjacency in enumerate(adjacency_list):
        if len(adjacency) > max_adjacency:
            max_adjacency = len(adjacency)
            max_vertex = i  # Vertices are 0-indexed
    return max_vertex, max_adjacency

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python program.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    adjacency_list = read_adjacency_file(filename)
    if adjacency_list is not None:
        max_vertex, max_adjacency = find_max_adjacency_vertex(adjacency_list)
        print(f"The vertex with the maximum adjacency is {max_vertex} with {max_adjacency} adjacencies.")


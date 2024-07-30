#include "../inc/common.h"

// DEBUG - MAX TRACKER VARIABLES
uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;
ofstream output_file;

// MPI VARIABLES
int wsize;
int grank;
char msg_buffer[NUMBER_OF_PROCESSESS][100];             // for every task there is a seperate message buffer and incoming/outgoing handle slot
MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];          // array of handles for messages with all other thread, allows for asynchronous messaging, handles say whether message is complete
MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
bool global_free_list[NUMBER_OF_PROCESSESS];

CPU_Graph::CPU_Graph(ifstream& graph_stream)
{
    graph_stream >> number_of_vertices;
    graph_stream >> number_of_edges;
    graph_stream >> number_of_lvl2adj;

    onehop_neighbors = new int[number_of_edges];
    onehop_offsets = new uint64_t[number_of_vertices + 1];
    twohop_neighbors = new int[number_of_lvl2adj];
    twohop_offsets = new uint64_t[number_of_vertices + 1];

    for (int i = 0; i < number_of_edges; i++) {
        graph_stream >> onehop_neighbors[i];
    }
    for (int i = 0; i < number_of_vertices + 1; i++) {
        graph_stream >> onehop_offsets[i];
    }
    // NOTE - making i a uint64_t here breaks the code even though it is being compared to a uint64_t
    for (int i = 0; i < number_of_lvl2adj; i++) {
        graph_stream >> twohop_neighbors[i];
    }
    for (int i = 0; i < number_of_vertices + 1; i++) {
        graph_stream >> twohop_offsets[i];
    }
}

CPU_Graph::~CPU_Graph() 
{
    delete onehop_neighbors;
    delete onehop_offsets;
    delete twohop_neighbors;
    delete twohop_offsets;
}

DS_Sizes::DS_Sizes(const string& filename)
{
    ifstream file(filename);
    string line;
    int line_count = 0;
    
    while (getline(file, line)) {

        size_t commaPos = line.find(',');
        if (commaPos != string::npos) {
            string valueStr = line.substr(commaPos + 1);
            uint64_t value = stoull(valueStr);

            switch (line_count) {
                case 0: tasks_size = value; break;
                case 1: tasks_per_warp = value; break;
                case 2: buffer_size = value; break;
                case 3: buffer_offset_size = value; break;
                case 4: cliques_size = value; break;
                case 5: cliques_offset_size = value; break;
                case 6: cliques_percent = value; break;
                case 7: wcliques_size = value; break;
                case 8: wcliques_offset_size = value; break;
                case 9: wtasks_size = value; break;
                case 10: wtasks_offset_size = value; break;
                case 11: wvertices_size = value; break;
            }
        }

        line_count++;
    }

    file.close();

    expand_threshold = (tasks_per_warp * NUMBER_OF_WARPS);
    cliques_dump = (cliques_size * (cliques_percent / 100.0));
}
#include "./inc/common.h"
#include "./inc/host_functions.h"
#include "./inc/host_debug.h"
#include "./inc/Quick_rmnonmax.h"

// TODO
// - see whether it's possible to parallelize some of calculate_LU_bounds

// CURSOR - 

// MAIN
int main(int argc, char* argv[])
{
    // DEBUG - rm
    cout << HELP_MULTIPLIER << endl;

    // TIME
    auto start2 = chrono::high_resolution_clock::now();

    double minimum_degree_ratio;        // connection requirement for cliques
    int minimum_clique_size;            // minimum size for cliques
    int* minimum_degrees;               // stores the minimum connections per vertex for all size cliques
    int world_size;                     // number of cpu processes
    int world_rank;                     // current cpu processes rank
    string filename;                    // used in concatenation for making filenames
    string filename2;                   // used for second filename in remove non max
    ifstream read_file;                 // multiple read files
    ofstream write_file;                // writing results to mutiple files
    string line;                        // stores lines from read file
    string output;                      // distinct output name so programs can be run simultaneously

    // ENSURE PROPER USAGE
    if (argc != 6) {
        printf("Usage: ./main <graph_file> <gamma> <min_size> <ds_sizes_file> <output_file>\n");
        return 1;
    }
    read_file.open(argv[4], ios::in);
    if(!read_file.is_open()){
        cout << "invalid data structure sizes file\n" << endl;
    }
    read_file.close();
    // reads the sizes of the data structures
    DS_Sizes dss(argv[4]);
    read_file.open(argv[1], ios::in);
    if (!read_file.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }
    minimum_degree_ratio = atof(argv[2]);
    if (minimum_degree_ratio < .5 || minimum_degree_ratio>1) {
        printf("minimum degree ratio must be between .5 and 1 inclusive\n");
        return 1;
    }
    minimum_clique_size = atoi(argv[3]);
    if (minimum_clique_size <= 1) {
        printf("minimum size must be greater than 1\n");
        return 1;
    }
    if (CPU_EXPAND_THRESHOLD > dss.expand_threshold) {
        cout << "CPU_EXPAND_THRESHOLD must be less than the EXPAND_THRESHOLD" << endl;
        return 1;
    }

    // MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    wsize = world_size;
    grank = world_rank;

    // DEBUG
    output = argv[5];
    filename = "o_" + output + "_" + to_string(grank) + ".txt";
    output_file.open(filename);
    if (DEBUG_TOGGLE) {
        output_file << endl << ">:OUTPUT FROM PROCESS: " << grank << endl << endl;
        initialize_maxes();
    }

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    if(grank == 0){
        cout << ">:PRE-PROCESSING" << endl;
    }
    CPU_Graph hg(read_file);
    read_file.close();
    calculate_minimum_degrees(hg, minimum_degrees, minimum_degree_ratio);
    filename = "t_" + output + "_" + to_string(grank) + ".txt";
    write_file.open(filename);

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    if(grank == 0){
        cout << "--->:LOADING TIME: " << duration.count() << " ms" << endl;
    }

    // SEARCH
    search(hg, write_file, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

    write_file.close();

    // DEBUG
    if (DEBUG_TOGGLE) {
        print_maxes();
    }
    output_file.close();

    // TIME
    auto start1 = chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
    if(grank == 0){
        // COMBINE RESULTS
        filename = "t_" + output + ".txt";
        write_file.open(filename);
        for (int i = 0; i < NUMBER_OF_PROCESSESS; ++i) {
            filename = "t_" + output + "_" + to_string(i) + ".txt";
            read_file.open(filename);
            while (getline(read_file, line)) {
                write_file << line << endl;
            }
            read_file.close();
        }

        // RM NON-MAX
        if(!(write_file.tellp() == ofstream::pos_type(0))){
            filename = "t_" + output + ".txt";
            filename2 = "r_" + output + ".txt";
            RemoveNonMax(filename.c_str(), filename2.c_str());
        }
        else{
            cout << ">:NUMBER OF MAXIMAL CLIQUES: 0" << endl;
        }
        write_file.close();
    }

    // TIME
    auto stop1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(stop1 - start1);
    if(grank == 0){
        cout << "--->:REMOVE NON-MAX TIME: " << duration1.count() << " ms" << endl;
    }
    auto stop2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(stop2 - start2);
    if(grank == 0){
        cout << "--->:TOTAL TIME: " << duration2.count() << " ms" << endl;
        cout << ">:PROGRAM END" << endl;
    }

    MPI_Finalize();
    return 0;
}
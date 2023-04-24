// MPI scan function
// HPC HW4 Q3

#include <mpi.h>
#include <iostream>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);
    
    long N = atol(argv[1]);

    double* local_scan = (double*) malloc(N/mpisize * sizeof(double));
    if (mpirank == 0) {
        double* scan_array = (double*) malloc(N * sizeof(double));
        for (long i = 0; i < N; ++i) scan_array[i] = rand()*2;

        MPI_Scatter(scan_array, N/mpisize, MPI_DOUBLE, &local_scan, N/mpisize, MPI_DOUBLE, 0, comm);
        free(scan_array);
    }

    
    double offset;
    for (long i = 1; i < N/mpisize; ++i) {
        local_scan[i] += local_scan[i-1];
        offset = local_scan[i];
    }

    MPI_Barrier(comm);
    double* all_offsets = (double*) malloc(mpisize*sizeof(double));
    MPI_Allgather(&offset, 1, MPI_DOUBLE, all_offsets, mpisize, MPI_DOUBLE, comm);

    if (mpirank != 0) {
        for (long i = 0; i < N/mpisize; ++i) {
            for (long j = 0; j < mpirank; ++j) {
                local_scan[i] += all_offsets[j];
            }
        }
    }

    MPI_Barrier(comm);
    double* final_array = (double*) malloc(N * sizeof(double));
    MPI_Gather(local_scan, N/mpisize, MPI_DOUBLE, final_array, N, MPI_DOUBLE, 0, comm);
    if (mpirank == 0) {
        std::cout << "Final scan = " << final_array[N-1] << std::endl;
    }
    free(final_array);
    free(local_scan);
    free(all_offsets);
    return 0;
}
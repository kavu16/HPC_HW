// integer ring MPI program
// HPC HW4 - Q2

#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpirank, mpisize;
    MPI_Comm_rank(comm, &mpirank);
    MPI_Comm_size(comm, &mpisize);

    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &namelen);
    std::cout << "Greene cluster id for process " << mpirank << ": " << processor_name << std::endl;

    long N = atol(argv[1]);

    MPI_Status status;
    double tt = MPI_Wtime();
    for (long i = 0; i < N; ++i) {
        if (mpirank == 0 && i == 0) {
            // 2MB = 2 * 2^20 bytes = 2^2 bytes * 2^19 bytes = sizeof(int) * 524288
            int* init = (int*) malloc(524288 * sizeof(int));
            for (long i = 0; i < 524288; ++i) init[i] = 0;
            // int init = 0;
            MPI_Send(&init, 1, MPI_INT, 1, 999, comm);
            // MPI_Send(init, 1, MPI_INT, 1, 999, comm);
            free(init);
        }
        else {
            int* curr;
            // int curr;
            MPI_Recv(&curr, 1, MPI_INT, (mpirank - 1 + mpisize)%mpisize, 999, comm, &status);
            // MPI_Recv(curr, 1, MPI_INT, (mpirank - 1 + mpisize)%mpisize, 999, comm, &status);

            // curr += mpirank;
            for (long i = 0; i < 524288; ++i) curr[i] += mpirank;

            MPI_Send(&curr, 1, MPI_INT, (mpirank + 1)%mpisize, 999, comm);
            // MPI_Send(curr, 1, MPI_INT, (mpirank + 1)%mpisize, 999, comm);
            free(curr);
        }
    }
    tt = MPI_Wtime() - tt;

    if (mpirank == 0) {
        // int final_sum;
        int* final_sum;
        MPI_Recv(&final_sum, 1, MPI_INT, mpisize-1, 999, comm, &status);
        // MPI_Recv(final_sum, 1, MPI_INT, mpisize-1, 999, comm, &status);

        int expected = 0;
        for (long i = 0; i < mpisize; ++i) {
            expected+=i;
        }
        expected*=N;

        std::cout<<"The expected sum was: "<<expected<<std::endl;
        std::cout<<"The final sum was: "<<final_sum[0]<<std::endl;
        std::cout<<"Message latency: "<<tt/N*1000<<" ms"<<std::endl;
    }

    MPI_Finalize();

}
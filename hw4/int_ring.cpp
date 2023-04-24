// integer ring MPI program
// HPC HW4 - Q2

#include <mpi.h>
#include <vector>
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
    for (long i = 0; i < N; ++i) {
        if (mpirank == 0 && i == 0) {
            int init = 0;
            MPI_Send(&init, 1, MPI_INT, 1, 999, comm);
        }
        else {
            int curr;
            MPI_Recv(&curr, 1, MPI_INT, (mpirank - 1 + mpisize)%mpisize, 999, comm, &status);

            curr += mpirank;

            MPI_Send(&curr, 1, MPI_INT, (mpirank + 1)%mpisize, 999, comm);
        }
    }

    if (mpirank == 0) {
        int final_sum;
        MPI_Recv(&final_sum, 1, MPI_INT, mpisize-1, 999, comm, &status);

        int expected = 0;
        for (long i = 0; i < mpisize; ++i) {
            expected+=i;
        }
        expected*=N;

        std::cout<<"The expected sum was: "<<expected<<std::endl;
        std::cout<<"The final sum was: "<<final_sum<<std::endl;
    }

    MPI_Finalize();

}
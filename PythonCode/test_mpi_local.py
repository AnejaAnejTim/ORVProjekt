from mpi4py import MPI
import sys

def test_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from Rank {rank} of {size} on {MPI.Get_processor_name()}")

if __name__ == "__main__":
    test_mpi()

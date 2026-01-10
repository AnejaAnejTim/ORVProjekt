import os
import sys
from mpi4py import MPI
from decompress_lib import decompress_file

def main():
    if len(sys.argv) < 3:
        print("Usage: mpiexec -n <N> python mpi_decompress.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"[Rank 0] MPI cluster started with {size} ranks.")

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        files = [f for f in os.listdir(input_dir) if f.endswith('.bin')]
        files.sort()
        print(f"[Rank {rank}] Found {len(files)} files in {input_dir}")
        if len(files) == 0:
            print(f"[Rank {rank}] No .bin files found to process.")
    else:
        files = None

    # Broadcast file list to all ranks
    files = comm.bcast(files, root=0)

    if not files:
        return

    my_files = files[rank::size]
    print(f"[Rank {rank}] My share of files: {my_files}")
    
    import time
    start_time = time.time()
    
    for filename in my_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".bmp"
        output_path = os.path.join(input_dir, output_filename)
        
        try:
            print(f"[Rank {rank}] Starting decompression: {filename}")
            decompress_file(input_path, output_path)
            print(f"[Rank {rank}] Finished: {filename} -> {output_filename}")
        except Exception as e:
            print(f"[Rank {rank}] Error decompressing {filename}: {e}")

    end_time = time.time()
    print(f"[Rank {rank}] Processed {len(my_files)} files in {end_time - start_time:.2f} seconds.")

    comm.Barrier()
    if rank == 0:
        print(f"[Rank {rank}] All decompression tasks completed.")

if __name__ == "__main__":
    if "OMPI_COMM_WORLD_SIZE" not in os.environ and "PMI_SIZE" not in os.environ:
        pass

    main()

from numba import cuda
import numpy

# CUDA notes:
# Block: a group of threads that execute the same code in parallel; maximum 1024 threads per block, threads have shared memory
# Grid: a group of blocks, each block contains the same number of threads
# OpenCL variant: https://pypi.org/project/pyopencl/
# PyCUDA: https://pypi.org/project/pycuda/
# Numba: https://numba.readthedocs.io/en/stable/
#   Example: https://nyu-cds.github.io/python-numba/05-cuda/

# RTX 4070 has compute level 8.9
# Threads per SM: 1536, SM count: 46, max threads per block: 1024, max active grids per device: 128
# Recommended thread/block count: 512 (3 blocks per SM)
# Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
print(cuda.gpus)
device = cuda.get_current_device()
print (device.compute_capability)

@cuda.jit
# Idea: find the prime candidate to check, run the computation, and set it to 0 if not prime
def my_kernel(io_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2 # do the computation

@cuda.jit
def is_prime(candidates, known_primes):
    # Compute flattened index inside the array
    pos = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if pos < candidates.size:  # Check array boundaries
        candidate = candidates[pos]
        for prime in known_primes:
            if candidate % prime == 0:
                candidates[pos] = 0  # Set to 0 if not prime
                break
            if prime > candidate // 2:
                break

# Create the data array - usually initialized some other way
data = numpy.ones(1024)
device_array = cuda.to_device(data)

# Set the number of threads in a block
threadsperblock = 256

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}")

# Now start the kernel
my_kernel[blockspergrid, threadsperblock](device_array)

# Copy the result back to the host
data = device_array.copy_to_host()

# Print the result
print(data)
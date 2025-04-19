from numba import cuda
import time
import math

# RTX 4070 has compute level 8.9
# Details: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
# Note: Specific capabilities (like SM count) can't be queried via the API and must be hardcoded.
# For deployment on other GPUs, a dictionary of values can be maintained and queried based on the GPU name/model.

primes = [2]

SM_count = 46
threads_per_block = 128
blocks_per_grid = SM_count * 4
max_chunk_size = threads_per_block * blocks_per_grid

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

def main_multiprocessing(count):
    found_primes = 0
    minV = primes[-1]+1
    # in case minV is even, start from the next odd number
    if minV % 2 == 0:
        minV += 1
    while found_primes < count:
        maxV = min(primes[-1] * 2, minV + max_chunk_size)
        index = 0
        candidates = [0] * max_chunk_size
        for candidate in range(minV, maxV, 2):
            candidates[index] = candidate
            index += 1

        print(f"Searching for primes in range {minV} to {maxV}...")
        candidates_cuda = cuda.to_device(candidates)
        known_primes_cuda = cuda.to_device(primes)
        is_prime[blocks_per_grid, threads_per_block](candidates_cuda, known_primes_cuda)
        
        candidates = candidates_cuda.copy_to_host()
        new_primes = [candidate for candidate in candidates if candidate != 0]
        new_primes.sort()

        for p in new_primes:
            primes.append(int(p))
        found_primes += len(new_primes)
        # if maxV is even, that means maxV-1 was the last checked number; we can start with the next odd number
        # if maxV is odd, it means maxV has not yet been checked, so we need to start the next cycle with maxV
        # it also guarantees that minV is odd
        minV = maxV + 1 if maxV % 2 == 0 else maxV
    return found_primes

if __name__ == "__main__":
    # format of data in file: [2, 3, 5, 7, 11, ...]
    # read primes from file 
    with open("primes.txt", "r") as f:
        primes = eval(f.read())
    primes = sorted(set(primes))  # remove duplicates and sort
    known_prime_count = len(primes)
    print(f"Starting number of known prime numbers: {known_prime_count}")

    count = int(input("Number of primes to find (0 to stop, may find more than requested): "))
    while count > 0:
        start = math.floor(time.time()*1000)
        found = main_multiprocessing(count)
        end = math.floor(time.time()*1000)
        print(f"Execution time: {end - start} ms")
        print(f"New prime numbers found: {found}")
        
        # write results into a file
        with open("primes.txt", "w") as f:
            f.write(f"{primes}")
        
        count = int(input("Number of primes to find (0 to stop, may find more than requested): "))
    
    print(f"Total number of new primes: {len(primes) - known_prime_count}")
    print(f"Total prime numbers now available: {len(primes)}")
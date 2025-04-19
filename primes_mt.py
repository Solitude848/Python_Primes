from multiprocessing import Pool
import time
import math

primes = [2]

prc_limit = 20
max_chunk_size = 2000

def is_prime_worker(n, known_primes):
    """Worker function for multiprocessing."""
    is_prime = True
    for prime in known_primes:
        if n % prime == 0:
            is_prime = False
            break
        if prime > n // 2:
            break
    return n if is_prime else None

def main_multiprocessing(count):
    """Main function using multiprocessing."""
    found_primes = 0
    minV = primes[-1]+1
    # in case minV is even, start from the next odd number
    if minV % 2 == 0:
        minV += 1
    pool = Pool(prc_limit)
    while found_primes < count:
        maxV = min(primes[-1] * 2, minV + max_chunk_size)
        print(f"Searching for primes in range {minV} to {maxV}...")
        
        results = pool.starmap(is_prime_worker, [(n, primes) for n in range(minV, maxV, 2)])
        new_primes = [result for result in results if result is not None]
        new_primes.sort()
        found_primes += len(new_primes)
        primes.extend(new_primes)

        # if maxV is even, that means maxV-1 was the last checked number; we can start with the next odd number
        # if maxV is odd, it means maxV has not yet been checked, so we need to start the next cycle with maxV
        # it also guarantees that minV is odd
        minV = maxV + 1 if maxV % 2 == 0 else maxV
    return found_primes

# run the main function if this script is executed directly
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
# https://code.visualstudio.com/docs/python/python-tutorial
import threading
import time
import math

# a list to hold prime numbers
primes = [2]
threads = []

# a function to check if a number is prime
def is_prime(n):
    """Check if a number is prime."""
    is_prime = True
    first_half = True
    index = 0
    max = primes[-1] / 2
    # for candidate n, only need to check divisibility with primes up to n/2
    while is_prime and first_half:
        if primes[index] > max:
            first_half = False
        
        if n % primes[index] == 0:
            is_prime = False
        
        index += 1
    
    if is_prime:
        primes.append(n)
    return is_prime

# main function to run the program
# creates a copy of the primes list to pass to is_prime
def main_multithread(limit):
    """Main function to run the program."""
    for n in range(3, limit + 1, 2):  # check only odd numbers
        #thread = threading.Thread(target=is_prime, args=(n))
        thread = threading.Thread(target=is_prime, args=(n,))
        threads.append(thread)
        thread.start()

        if(n > primes[-1]*2 or len(threads) >= 20):
            # wait for all threads to finish before checking the next number
            for thread in threads:
                thread.join()
            threads.clear()
            primes.sort()
    primes.sort()

def main_singlethread(limit):
    """Main function to run the program."""
    for n in range(3, limit + 1, 2):  # check only odd numbers
        is_prime(n)

# run the main function if this script is executed directly
if __name__ == "__main__":
    limit = int(input("Enter a limit: "))
    start = math.floor(time.time()*1000)
    main_multithread(limit)
    # main_singlethread(limit)
    end = math.floor(time.time()*1000)
    print(f"Execution time: {end - start} ms")
    print(f"Total prime numbers found: {len(primes)}")
    # write results into a file
    with open("primes.txt", "w") as f:
        f.write(f"{primes}")
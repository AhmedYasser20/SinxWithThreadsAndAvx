### GitHub README for `SinxWithThreadsAndAvx.cpp`

---

# SinxWithThreadsAndAvx

This project demonstrates the calculation of the sine function using multi-threading and AVX (Advanced Vector Extensions) for performance optimization. The goal is to compute an approximation of the sine function for a given set of input values, using both threading and SIMD (Single Instruction, Multiple Data) optimizations.

## Features

- **Multithreading (Pthreads):** Utilizes multiple threads to divide the workload and compute sine values in parallel.
- **AVX Optimizations:** Uses AVX instructions to compute sine values on 8 floats simultaneously, speeding up the computation.
- **Comparison of Execution Times:** Compares execution times between:
  - Threads only
  - Threads + AVX
  - AVX only

## Requirements

- GCC compiler with support for AVX and pthread.
- Linux-based system with AVX hardware support.
- Aligned memory for AVX vectorization.

## Compilation

To compile the code, use the following command:

```bash
g++ -mavx -pthread SinxWithThreadsAndAvx.cpp -o SinxWithThreadsAndAvx
```

## Execution

Run the executable as follows:

```bash
./SinxWithThreadsAndAvx
```

The output will show the computation time for the different approaches.

### Example Output:

```
N (number of terms): 128, Terms: 10000, Number of Threads: 32
Threads Only Time: 10.802ms
Threads And AVX Time: 5.998ms
AVX Only Time: 2.345ms
```

## Code Structure

- `sinx_mtWithAVX`: Combines threading and AVX instructions for fast sine computation.
- `sinx_mt`: Multithreaded sine computation without AVX.
- `sinx`: AVX-only sine computation.
- **ThreadData**: Struct for passing data to threads.
- **sinx_thread2** and **sinx_thread**: Functions executed by the threads to calculate sine values.




#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <algorithm>
#include <pthread.h>
using namespace std;
typedef struct {
    int start;
    int end;
    int terms;
    float* x;
    float* result;
} ThreadData;

void* sinx_thread2(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end;i += 8) {
         __m256 origx = _mm256_load_ps(&data->x[i]);  // We can use load_ps since we ensure alignment
        __m256 value = origx;
        __m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
        __m256 denom = _mm256_set1_ps(6.0f);  // 3!
        __m256 sign = _mm256_set1_ps(-1.0f);

        for (int j = 1; j <= data->terms; j++)
        {
            __m256 tmp = _mm256_div_ps(_mm256_mul_ps(sign, numer), denom);
            value = _mm256_add_ps(value, tmp);
            numer = _mm256_mul_ps(numer, _mm256_mul_ps(origx, origx));
            denom = _mm256_mul_ps(denom, _mm256_set1_ps((2 * j + 2) * (2 * j + 3)));
            sign = _mm256_mul_ps(sign, _mm256_set1_ps(-1.0f));
        }
        
        _mm256_store_ps(&data->result[i], value); 
    }
    return NULL;
}

void sinx_mtWithAVX(int N, int terms, float* x, float* result, int num_threads) {
    clock_t starttime, endtime;

    starttime = clock();
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Ensure chunk size is a multiple of 8
    int chunk_size = (N / num_threads + 7) / 8 * 8; // Round up to nearest multiple of 8
    int start = 0;
    int end = 0;

    for (int i = 0; i < num_threads; i++) {
        end = start + chunk_size;
        
        // Make sure that the last thread doesn't exceed N
        if (end > N) end = N;

        thread_data[i].start = start;
        thread_data[i].end = end;
        thread_data[i].terms = terms;
        thread_data[i].x = x;
        thread_data[i].result = result;

        pthread_create(&threads[i], NULL, sinx_thread2, &thread_data[i]);

        start = end;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    endtime = clock();
    cout << "Threads And AVX Time : " << ((double)(endtime - starttime)) / CLOCKS_PER_SEC * 1000.0 << "ms" << endl;
}



void* sinx_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; i++) {
        float value = data->x[i];
        float numer = data->x[i] * data->x[i] * data->x[i];
        int denom = 6;  // 3!
        int sign = -1;

        for (int j = 1; j <= data->terms; j++) {
            value += sign * numer / denom;
            numer *= data->x[i] * data->x[i];
            denom *= (2*j+2) * (2*j+3);
            sign *= -1;
        }

        data->result[i] = value;
    }
    return NULL;
}
void sinx_mt(int N, int terms, float* x, float* result, int num_threads) {
     clock_t starttime, endtime;

     starttime = clock();
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int chunk_size = N / num_threads;
    int remainder = N % num_threads;
    int start = 0;

    for (int i = 0; i < num_threads; i++) {
        int end = start + chunk_size + (i < remainder ? 1 : 0);
        
        thread_data[i].start = start;
        thread_data[i].end = end;
        thread_data[i].terms = terms;
        thread_data[i].x = x;
        thread_data[i].result = result;

        pthread_create(&threads[i], NULL, sinx_thread, &thread_data[i]);

        start = end;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    endtime = clock();
    cout<<"Threads Only Time : "<<((double) (endtime - starttime)) / CLOCKS_PER_SEC * 1000.0 <<"ms"<<endl;
}



void sinx(int N, int terms, float *x, float *result)
{
    clock_t starttime = clock();
    
    for (int i = 0; i < N; i += 8)
    {
        __m256 origx = _mm256_load_ps(&x[i]);  // We can use load_ps since we ensure alignment
        __m256 value = origx;
        __m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
        __m256 denom = _mm256_set1_ps(6.0f);  // 3!
        __m256 sign = _mm256_set1_ps(-1.0f);

        for (int j = 1; j <= terms; j++)
        {
            __m256 tmp = _mm256_div_ps(_mm256_mul_ps(sign, numer), denom);
            value = _mm256_add_ps(value, tmp);
            numer = _mm256_mul_ps(numer, _mm256_mul_ps(origx, origx));
            denom = _mm256_mul_ps(denom, _mm256_set1_ps((2 * j + 2) * (2 * j + 3)));
            sign = _mm256_mul_ps(sign, _mm256_set1_ps(-1.0f));
        }
        
        _mm256_store_ps(&result[i], value);  // We can use store_ps since we ensure alignment
    }

    clock_t endtime = clock();
    cout << "AVX Only Time: " << ((double)(endtime - starttime)) / CLOCKS_PER_SEC * 1000.0 << "ms" << endl;
}

int main(int argc, char **argv)
{
    const int N = 128;  // Changed to 16
    const int terms = 10000;
    int num_threads = 32;
    // Allocate aligned memory for both x and result
    float* x = (float*)aligned_alloc(32, N * sizeof(float));
    float* result = (float*)aligned_alloc(32, N * sizeof(float));
    
    if (x == nullptr || result == nullptr) {
        cerr << "Memory allocation failed" << endl;
        return 1;
    }

    // Initialize x with some values
    for (int i = 0; i < N; i++) {
        x[i] = (i % 5 == 0) ? 1.2f : 
               (i % 5 == 1) ? 1.0f : 
               (i % 5 == 2) ? 90.0f : 
               (i % 5 == 3) ? 2.2f : 5.54f;
    }

    cout<<"N (number of term) : "<< N<< " Terms : "<<terms<<" Number of Threads : "<<num_threads<<endl;
    sinx_mt(N, terms, x, result, num_threads);
    sinx_mtWithAVX(N, terms, x, result, num_threads);
    sinx(N, terms, x, result);

   

    free(x);
    free(result);

    return 0;
}
/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <unistd.h>
#include <cmath>
#include <sys/time.h>
#include <nv_gpu_cache.hpp>
#include <omp.h>

// The random number generator
template<typename T>
class IntGenerator {
public:
    IntGenerator(): gen_(rd_()) {}
    IntGenerator(T min, T max): gen_(rd_()), distribution_(min, max) {}

    void fill_unique(T* data, size_t len, T empty_value) {
        if (len == 0) {
            return;
        }
        assert(distribution_.max() - distribution_.min()>= len);

        std::unordered_set<T> set;
        size_t sz = 0;
        while (sz < len) {
            T x = distribution_(gen_);
            if(x == empty_value){
                continue;
            }
            auto res = set.insert(x);
            if (res.second) {
                data[sz++] = x;
            }
        }
        assert(sz == set.size());
        assert(sz == len);
    }

private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> distribution_;
};

template<typename T>
class IntGenerator_normal {
public:
    IntGenerator_normal(): gen_(rd_()) {}
    IntGenerator_normal(double mean, double dev): gen_(rd_()), distribution_(mean, dev) {}

    void fill_unique(T* data, size_t len, T min, T max) {
        if (len == 0) {
            return;
        }

        std::unordered_set<T> set;
        size_t sz = 0;
        while (sz < len) {
            T x = (T)(abs( distribution_(gen_) ));
            if(x < min || x > max){
                continue;
            }
            auto res = set.insert(x);
            if (res.second) {
                data[sz++] = x;
            }
        }
        assert(sz == set.size());
        assert(sz == len);
    }

private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<double> distribution_;
};

// Utility to fill len embedding vector
template<typename KeyType>
void fill_vec(const KeyType* keys, float* vals, size_t embedding_vec_size, size_t len, float ratio) {
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < embedding_vec_size; ++j) {
            vals[i * embedding_vec_size + j] = (float)(ratio * keys[i]);
        }
    }
}

// Floating-point compare function
template<typename T>
bool is_near(T a, T b) {
    double diff = abs(a - b);
    bool ret = diff <= std::min(a, b) * 1e-6;
    if (!ret) {
        std::cerr << "error: " << a << " != " << b << "; diff = " << diff << std::endl;
    }
    return ret;
}

// Check correctness of result
template<typename KeyType>
void check_result(const KeyType* keys, const float* vals, size_t embedding_vec_size, size_t len, float ratio) {
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < embedding_vec_size; ++j) {
            assert(is_near(vals[i * embedding_vec_size + j], (float)(ratio * keys[i])));
        }
    }
}

/* Timing funtion */
double W_time(){
    timeval marker;
    gettimeofday(&marker, NULL);
    return ((double)(marker.tv_sec) * 1e6 + (double)(marker.tv_usec)) * 1e-6;
}

using key_type = long long;
using ref_counter_type = uint64_t;

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0] << " embedding_table_size cache_capacity_in_set embedding_vec_size query_length iter_round num_threads" << std::endl;
        return -1;
    }

    // Arguments
    const size_t emb_size = atoi(argv[1]);
    const size_t cache_capacity_in_set = atoi(argv[2]);
    const size_t embedding_vec_size = atoi(argv[3]);
    const size_t query_length = atoi(argv[4]);
    const size_t iter_round = atoi(argv[5]);
    const size_t num_threads = atoi(argv[6]);

    // Since cache is designed for single-gpu, all threads just use GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // Host side buffers shared between threads
    key_type* h_keys; // Buffer holding all keys in embedding table
    float* h_vals; // Buffer holding all values in embedding table

    // host-only buffers placed in normal host memory
    h_keys = (key_type*)malloc(emb_size * sizeof(key_type));
    h_vals = (float*)malloc(emb_size * embedding_vec_size * sizeof(float));

    // The empty key to be used
    const key_type empty_key = std::numeric_limits<key_type>::max();

    // The cache to be used, by default the set hasher is based on MurMurHash and slab hasher is based on Mod.
    using Cache_ = gpu_cache::gpu_cache<key_type, ref_counter_type, empty_key, SET_ASSOCIATIVITY, SLAB_SIZE>; 

    // Create GPU embedding cache
    auto cache = new Cache_(cache_capacity_in_set, embedding_vec_size);

    // For random unique keys generation 
    IntGenerator<key_type> gen_key;
    float increase = 0.1;

    std::cout << "****************************************" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Start Single-GPU Thread-safe Cache test " << std::endl;

    // Timimg variables
    double time_a;
    double time_b;

    time_a = W_time();

    std::cout << "Filling data" << std::endl;
    // Generating random unique keys
    gen_key.fill_unique(h_keys, emb_size, empty_key);
    // Filling values vector according to the keys
    fill_vec(h_keys, h_vals, embedding_vec_size, emb_size, increase);

    // Elapsed wall time
    time_b = W_time() - time_a;
    std::cout << "The Elapsed time for filling data is: " << time_b << "sec." << std::endl;

    // Insert <k,v> pairs to embedding table (CPU hashtable)
    time_a = W_time();

    std::cout << "Filling embedding table" << std::endl;
    std::unordered_map<key_type, std::vector<float>> h_emb_table;
    for(size_t i = 0; i < emb_size; i++){
        std::vector<float> emb_vec(embedding_vec_size);
        for(size_t j = 0; j < embedding_vec_size; j++){
            emb_vec[j] = h_vals[i * embedding_vec_size + j];
        }
        h_emb_table.emplace(h_keys[i], emb_vec);
    }

    // Elapsed wall time
    time_b = W_time() - time_a;
    std::cout << "The Elapsed time for filling embedding table is: " << time_b << "sec." << std::endl;

    // Free value buffer
    free(h_vals);


#pragma omp parallel default(none) shared(h_keys, cache, h_emb_table, increase, embedding_vec_size, query_length, emb_size, iter_round) num_threads(num_threads)
{
    // The thread ID for this thread
    int thread_id = omp_get_thread_num();
    printf("Worker %d starts testing cache.\n", thread_id);
    // Since cache is designed for single-gpu, all threads just use GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // Thread-private host side buffers
    size_t* h_query_keys_index; // Buffer holding index for keys to be queried
    key_type* h_query_keys; // Buffer holding keys to be queried
    float* h_vals_retrieved; // Buffer holding values retrieved from query
    key_type* h_missing_keys; // Buffer holding missing keys from query
    float* h_missing_vals; // Buffer holding values for missing keys
    uint64_t* h_missing_index; // Buffers holding missing index from query
    size_t h_missing_len; // missing length

    // Thread-private device side buffers
    key_type* d_query_keys; // Buffer holding keys to be queried
    float* d_vals_retrieved; // Buffer holding values retrieved from query
    key_type* d_missing_keys; // Buffer holding missing keys from query
    float* d_missing_vals; // Buffer holding values for missing keys
    uint64_t* d_missing_index; // Buffers holding missing index from query
    size_t* d_missing_len; // missing length

    // host-only buffers placed in normal host memory
    h_query_keys_index = (size_t*)malloc(query_length * sizeof(size_t));
    // host-device interactive buffers placed in pinned memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_query_keys, query_length * sizeof(key_type), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_vals_retrieved, query_length * embedding_vec_size * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_keys, query_length * sizeof(key_type), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_vals, query_length * embedding_vec_size * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_index, query_length * sizeof(uint64_t), cudaHostAllocPortable));

    // Allocate device side buffers
    CUDA_CHECK(cudaMalloc((void**)&d_query_keys, query_length * sizeof(key_type)));
    CUDA_CHECK(cudaMalloc((void**)&d_vals_retrieved, query_length * embedding_vec_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_keys, query_length * sizeof(key_type)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_vals, query_length * embedding_vec_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_index, query_length * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_len, sizeof(size_t)));

    // Thread-private CUDA stream, all threads just use the #0 device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Timimg variables
    double time_1;
    double time_2;

    /******************************************************************************
    *******************************************************************************
    ********************************Test start*************************************
    *******************************************************************************
    *******************************************************************************/

    // Normal-distribution random number generator
    size_t foot_print = emb_size - 1; // Memory footprint for access the keys within the key buffer
    double mean = (double)(emb_size / 2); // mean for normal distribution
    double dev = (double)(2 * query_length); // dev for normal distribution
    // IntGenerator<size_t> uni_gen(0, foot_print);
    // Normal-distribution random number generator
    IntGenerator_normal<size_t> normal_gen(mean, dev);

    // Start normal distribution cache test
    printf("Worker %d : normal distribution test start.\n", thread_id);
    for(size_t i = 0; i < iter_round; i++){

        // Generate random index with normal-distribution
        normal_gen.fill_unique(h_query_keys_index, query_length, 0, foot_print);
        // Select keys from total keys buffer with the index
        for(size_t j = 0; j < query_length; j++){
            h_query_keys[j] = h_keys[h_query_keys_index[j]];
        }

        // Copy the keys to GPU memory
        CUDA_CHECK(cudaMemcpyAsync(d_query_keys, h_query_keys, query_length * sizeof(key_type), cudaMemcpyHostToDevice, stream));
        // Wait for stream to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Record time
        time_1 = W_time();
        // Get pairs from hashtable
        cache -> Query(d_query_keys, query_length, d_vals_retrieved, d_missing_index, d_missing_keys, d_missing_len, stream);
        // Wait for stream to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Elapsed wall time
        time_2 = W_time() - time_1;
        printf("Worker %d : The Elapsed time for %zu round normal-distribution query is: %f sec.\n", thread_id, i, time_2);

        // Copy the data back to host
        CUDA_CHECK(cudaMemcpyAsync(h_vals_retrieved, d_vals_retrieved, query_length * embedding_vec_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_missing_index, d_missing_index, query_length * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_missing_keys, d_missing_keys, query_length * sizeof(key_type), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_missing_len, d_missing_len, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("Worker %d : %zu round : Missing key: %zu. Hit rate: %f %%.\n", thread_id, i, h_missing_len, 100.0f - (((float)h_missing_len / (float)query_length) * 100.0f));

        // Get missing key from embedding table
        // Insert missing values into the retrieved value buffer
        // Record time
        time_1 = W_time();
        for(size_t missing_idx = 0; missing_idx < h_missing_len; missing_idx++){
            auto result = h_emb_table.find(h_missing_keys[missing_idx]);
            for(size_t emb_vec_idx = 0; emb_vec_idx < embedding_vec_size; emb_vec_idx++){
                h_missing_vals[missing_idx * embedding_vec_size + emb_vec_idx] = (result->second)[emb_vec_idx];
                h_vals_retrieved[h_missing_index[missing_idx] * embedding_vec_size + emb_vec_idx] = (result->second)[emb_vec_idx];
            }
        }
        // Elapsed wall time
        time_2 = W_time() - time_1;
        printf("Worker %d : The Elapsed time for %zu round normal-distribution fetching missing data is: %f sec.\n", thread_id, i, time_2);

        // Copy the missing value to device
        CUDA_CHECK(cudaMemcpyAsync(d_missing_vals, h_missing_vals, query_length * embedding_vec_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Record time
        time_1 = W_time();
        // Replace the missing <k,v> pairs into the cache
        cache -> Replace(d_missing_keys, h_missing_len, d_missing_vals, stream);
        // Wait for stream to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Elapsed wall time
        time_2 = W_time() - time_1;
        printf("Worker %d : The Elapsed time for %zu round normal-distribution replace is: %f sec.\n", thread_id, i, time_2);

        // Verification: Check for correctness for retrieved pairs
        check_result(h_query_keys, h_vals_retrieved, embedding_vec_size, query_length, increase);
        printf("Worker %d : Result check for %zu round normal-distribution query+replace successfully!\n", thread_id, i);

    }

    printf("Worker %d : All Finished!\n", thread_id);

    // Clean-up
    cudaStreamDestroy(stream);
    free(h_query_keys_index);
    CUDA_CHECK(cudaFreeHost(h_query_keys));
    CUDA_CHECK(cudaFreeHost(h_vals_retrieved));
    CUDA_CHECK(cudaFreeHost(h_missing_keys));
    CUDA_CHECK(cudaFreeHost(h_missing_vals));
    CUDA_CHECK(cudaFreeHost(h_missing_index));

    CUDA_CHECK(cudaFree(d_query_keys));
    CUDA_CHECK(cudaFree(d_vals_retrieved));
    CUDA_CHECK(cudaFree(d_missing_keys));
    CUDA_CHECK(cudaFree(d_missing_vals));
    CUDA_CHECK(cudaFree(d_missing_index));
    CUDA_CHECK(cudaFree(d_missing_len));

}
    
    free(h_keys);
    delete cache;
    return 0;
}

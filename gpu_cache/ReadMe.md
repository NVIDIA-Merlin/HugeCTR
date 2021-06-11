# GPU embedding cache
This project implements a embedding cache on GPU memory designed for CTR inference workload.

The cache will store the hot [embedding id, embedding vectors] pairs on GPU memory, thus can reduce the trafic to parameter server when doing embedding table lookup.

The cache is desinged for CTR inference, it has following features/restrictions:
* The cache is read-only: modifying the embedding vector of a embedding id is not supported.
* All the backup memory side operation is performed by parameter server(prefetching, latency hiding etc.).
* Single-GPU design, each cache belongs to one GPU.
* The cache is thread-safe: multiple workers(CPU threads) can concurrently call the API of a single cache object with defined-behavior.
* The cache implements a LRU(Least-Recent-Use) replacement algorithm, so it will cache the most recently queried embeddings.
* The embeddings stored inside the cache is unique: no duplicated embedding id in the cache.

## Project structure
This project is a stand-alone module in HugeCTR project. The root folder of this project is the `gpu_cache` folder under the HugeCTR root directory. 

The `include` folder contains the headers for the cache library and the `src` folder contains the implementations and makefile for the cache library.

The `nv_gpu_cache.hpp` file contains the definition of the main classes: `gpu_cache` that implement the GPU embedding cache while the `nv_gpu_cache.cu` file contains the implementation.

As a module of HugeCTR, this project will be built with and utilized by the HugeCTR project. 

## Supported data types
* Currently the cache only support 32/64 bit scalar integer type for key(embedding id) type. For example, `unsigned int`, `long long` etc.
* Currently the cache only support float vector for value(embedding vector) type. 
* User need to give an empty key to indicate the empty bucket. User should never use empty key to represent any real key.
* User can refer to the instantiation code at the end of `nv_gpu_cache.cu` file for template parameters.

## Requirement
* NVIDIA GPU >= Volta(sm_70).
* CUDA environment >= 9.0.
* (Optional)libcu++ library >= 1.1.0. The CUDA Toolkit 11.0 and above already contains libcu++ 1.1.0. If user use CUDA Toolkit 10.2, stand-alone version of libcu++ library is required and can be found here: https://github.com/NVIDIA/libcudacxx. Please note that, libcu++ library only supports CUDA environment >= 10.2. If CUDA version is not a problem for the user, it is recommended to use the libcu++ library since it has higher performance and precisely-defined behavior. User can enable libcu++ library by defining `LIBCUDACXX_VERSION` macro when compiling. Otherwise, libcu++ library will not be used so CUDA 9.0-10.1 will be supported.
* The default building option in HugeCTR is to disable the libcu++ library. 

## Usage overview
```
template<typename key_type,
         typename ref_counter_type, 
         key_type empty_key, 
         int set_associativity, 
         int warp_size,
         typename set_hasher = MurmurHash3_32<key_type>, 
         typename slab_hasher = Mod_Hash<key_type, size_t>>
class gpu_cache{
public:
    //Ctor
    gpu_cache(const size_t capacity_in_set, const size_t embedding_vec_size);

    //Dtor
    ~gpu_cache();

    // Query API, i.e. A single read from the cache
    void Query(const key_type* d_keys, 
               const size_t len, 
               float* d_values, 
               uint64_t* d_missing_index, 
               key_type* d_missing_keys, 
               size_t* d_missing_len, 
               cudaStream_t stream);


    // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
    void Replace(const key_type* d_keys, 
                 const size_t len, 
                 const float* d_values, 
                 cudaStream_t stream);

};

```
`Constructor`

To create a new embedding cache, user need to provide:
* Template parameters: 
    + key_type: the data type of embedding id. 
    + ref_counter_type: the data type of the internal counter. This data type should be 64bit unsigned integer(i.e. uint64_t), 32bit integer has the risk of overflow. 
    + empty_key: the key value indicate for empty bucket(i.e. The empty key), user should never use empty key value to represent any real keys.
    + set_associativity: the hyper-parameter indicats how many slabs per cache set.(See `Performance hint` session below)
    + warp_size: the hyper-parameter indicats how many [key, value] pairs per slab. Acceptable value includes 1/2/4/8/16/32.(See `Performance hint` session below)
    + For other template parameters just use the default value.
* Parameters:
    + capacity_in_set: # of cache set in the embedding cache. So the total capacity of the embedding cache is `warp_size * set_associativity * capacity_in_set` [key, value] pairs.
    + embedding_vec_size: # of float per a embedding vector.
* The host thread will wait for the GPU kernels to complete before returning from the API, thus this API is synchronous with CPU thread. When returned, the initialization process of the cache is already done.
* The embedding cache will be created on the GPU where user call the constructor. Thus, user should set the host thread to the target CUDA device before creating the embedding cache. All resources(i.e. device-side buffers, CUDA streams) used later for this embedding cache should be allocated on the same CUDA device as the embedding cache.
* The constructor can be called only once, thus is not thread-safe.

`Destructor`
* The destructor clean up the embedding cache. This API should be called only once when user need to delete the embedding cache object, thus is not thread-safe.

`Query`
* Search `len` elements from device-side buffers `d_keys` in the cache and return the result in device-side buffer `d_values` if a key is hit in the cache.
* If a key is missing, the missing key and its index in the `d_keys` buffer will be returned in device-side buffers `d_missing_keys` and `d_missing_index`. The # of missing key will be return in device-side buffer `d_missing_len`. For simplicity, these buffers should have the same length as `d_keys` to avoid out-of-bound access.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* The keys to be queried in the `d_keys` buffer can have duplication. In this case, user will get duplicated returned values or missing information.
* This API is thread-safe and can be called concurrently with other `Query` and `Replace` APIs.

`Replace`
* The API will replace `len` [key, value] pairs listed in `d_keys` and `d_values` into the embedding cache using the LRU replacement algorithm.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* The keys to be replaced in the `d_keys` buffer can have duplication and can be already stored inside the cache. In these cases, the cache will detect any possible duplication and maintain the uniqueness of all the [key ,value] pairs stored in the cache.
* This API is thread-safe and can be called concurrently with other `Query` and `Replace` APIs.
* This API will first try to insert the [key, value] pairs into the cache if there is any empty slot. If the cache is full, it will do the replacement.

## More information
* The detailed introduction of the GPU embedding cache data structure is presented at GTC China 2020: https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20626-%e4%bd%bf%e7%94%a8+gpu+embedding+cache+%e5%8a%a0%e9%80%9f+ctr+%e6%8e%a8%e7%90%86%e8%bf%87%e7%a8%8b
* This project is used by `embedding_cache` class in `HugeCTR/include/inference/embedding_cache.hpp` which can be used as an example.

## Performance hint
* The hyper-parameter `warp_size` should be keep as 32 by default. When the length for Query or Replace operations is small(~1-50k), user can choose smaller warp_size and increase the total # of cache set(while maintaining the same cache size) to increase the parallelism and improve the performance.
* The hyper-parameter `set_associativity` is critical to performance: 
    + If set too small, may cause load imbalance between different cache sets(lower down the effective capacity of the cache, lower down the hit rate). To prevent this, the embedding cache uses a very random hash function to hash the keys to different cache set, thus will achieve load balance statistically. However, larger cache set will tends to have better load balance. 
    + If set too large, the searching space for a single key will be very large. The performance of the embedding cache API will drop dramatically. Also, each set will be accessed exclusively, thus the more cache sets the higher parallelism can be achieved.
    + Recommend setting `set_associativity` to 2 or 4.
* The GPU is designed for optimizing throughput. Always try to batch up the inference task and try to have larger `query_size`. 
* As the APIs of the embedding cache is asynchronous with host threads. Try to optimize the E2E inference pipeline by overlapping asynchronous tasks on GPU or between CPU and GPU. For example, after retrieving the missing values from the parameter server, user can combine the missing values with the hit values and do the rest of inference pipeline at the same time with the `Replace` API. Replacement is not necessarily happens together with Query all the time, user can do query multiple times then do a replacement if the hit rate is acceptable.
* Try different cache capacity and evaluate the hit rate. If the capacity of embedding cache can be larger than actual embedding footprint, the hit rate can be as high as 99%+.








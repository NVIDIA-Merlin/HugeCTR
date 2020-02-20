/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */


#pragma once

#include "FasterGossipCommMultiTraits.h"
#include "mpi.h"
#include <ucp/api/ucp.h>
#include <hwloc.h>
#include <hwloc/cudart.h>
#include <algorithm>
#include <omp.h>

#define WARM_UP_ROUND 2

namespace FasterGossipCommMulti{

// The empty call back function for UCP communication API
void empty_send_callback_func (void * request, ucs_status_t status) {}
void empty_recv_callback_func (void * request, ucs_status_t status, ucp_tag_recv_info_t *info) {}

template<typename data_t_, typename GossipMultiCommTraits>
class FasterGossipCommMulti;

template<typename data_t_>
class FasterGossipCommMulti<data_t_, FasterGossipCommMultiAll2AllTraits<data_t_>>{
public:
    using GossipMultiCommTraits = FasterGossipCommMultiAll2AllTraits<data_t_>;
    using FasterGossipComm = typename GossipMultiCommTraits::FasterGossipComm;
    using gpu_id_t = typename GossipMultiCommTraits::gpu_id_t;

    // Ctor
    FasterGossipCommMulti(const std::string& plan_file, const std::vector<gpu_id_t>& GPU_list, 
                          const int num_proc, const int rank, MPI_Comm comm) 
                          : GPU_list_(GPU_list), rank_(rank), num_proc_(num_proc), comm_(comm), GossipCommHandle_(num_proc_),
                          local_buffer_(GPU_list_.size()), recv_buffer_(GPU_list_.size()), temp_buf_(GPU_list_.size()),
                          temp_table_( GPU_list_.size() , std::vector<size_t>( GPU_list_.size() ) ),
                          temp_src_(GPU_list_.size()), temp_dst_(GPU_list_.size()), affinity_list_(num_proc_),
                          send_reqs_(GPU_list_.size(), nullptr), recv_reqs_(GPU_list_.size(), nullptr){
        // Do some check
        assert( (num_proc_ > 0) && "The number of process is not greater than 0!\n" );
        assert( (rank_ >= 0) && (rank_ < num_proc_) && "The rank of this process is not valid!\n" );
        // Local and total GPU count
        num_local_gpu_ = GPU_list_.size();
        num_total_gpu_ = num_proc_ * num_local_gpu_;
        assert( (num_local_gpu_ > 0) && "The number of local GPUs is not valid!\n" );
        // Create MPI_Request buffer
        //request_ = (MPI_Request* )malloc(2 * num_local_gpu_ * sizeof(MPI_Request));
        // Construct the local gossip all2all library
        for(int stage = 0; stage < num_proc_; stage++){

            GossipCommHandle_[stage] = new FasterGossipComm(plan_file, GPU_list_);

        }
        

        // HWLOC variable setup
        hwloc_topology_init(&topo_);
        hwloc_topology_set_io_types_filter(topo_, HWLOC_TYPE_FILTER_KEEP_ALL);
        hwloc_topology_load(topo_);
        hwloc_cpuset_t ori_cpu_set;
        hwloc_cpuset_t cpu_set;
        ori_cpu_set = hwloc_bitmap_alloc();
        cpu_set = hwloc_bitmap_alloc();

        // Get the original thread binding for recovery
        hwloc_get_cpubind(topo_, ori_cpu_set, HWLOC_CPUBIND_THREAD);

        // Get the number of CPU sockets and resize the UCP vector
        socket_num_ = hwloc_get_nbobjs_by_type(topo_, HWLOC_OBJ_PACKAGE);
        assert( (socket_num_ > 0) && "The number of CPU sockets is not valid!\n" );

        // Temp variable used to initialize UCP environment
        ucp_params_t ucp_params;
        ucp_config_t *ucp_config;
        ucp_worker_params_t ucp_worker_params;
        size_t ucp_worker_address_len;
        std::vector<ucp_ep_params_t> ucp_ep_params(socket_num_ * num_proc_);

        ucp_context_.resize(socket_num_);
        ucp_worker_.resize(socket_num_);
        ucp_worker_address_.resize(socket_num_);
        ucp_worker_address_book_.resize(socket_num_ * num_proc_);
        ucp_endpoints_.resize(socket_num_, std::vector<ucp_ep_h>(socket_num_ * num_proc_));

        // Initialize UCP Env on different CPU sockets
        for( int i = 0; i < socket_num_; i++){

            // Bind the current thread to run on target CPU socket
            hwloc_obj_t current_socket = hwloc_get_obj_by_type(topo_, HWLOC_OBJ_PACKAGE, i);
            hwloc_set_cpubind(topo_, current_socket->cpuset, HWLOC_CPUBIND_THREAD);

            // Test the place where the current thread is running
            hwloc_get_last_cpu_location(topo_, cpu_set, HWLOC_CPUBIND_THREAD);
            char * cpu_string;
            hwloc_bitmap_asprintf(&cpu_string, cpu_set);
            printf("On rank %d, the cpu set that current thread is running on is : %s.\n", rank_, cpu_string);
            free(cpu_string);

            // Initialize UCP context
            memset(&ucp_params, 0, sizeof(ucp_params));
            ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
            ucp_params.features          = UCP_FEATURE_TAG;
            ucp_params.estimated_num_eps = socket_num_ * num_proc_;

            ucp_config_read(NULL, NULL, &ucp_config);

            ucp_init(&ucp_params, ucp_config, &ucp_context_[i]);

            ucp_config_release(ucp_config);

            // Initialize UCP worker
            memset(&ucp_worker_params, 0, sizeof(ucp_worker_params));

            ucp_worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
            ucp_worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;  // only single thread can access this worker at one time, i.e. no thread safety.
            ucp_worker_create(ucp_context_[i], &ucp_worker_params, &ucp_worker_[i]);

            // Get address for local worker
            ucp_worker_get_address(ucp_worker_[i], &ucp_worker_address_[i], &ucp_worker_address_len);

        }

        // Recover the CPU binding of current thread
        hwloc_set_cpubind(topo_, ori_cpu_set, HWLOC_CPUBIND_THREAD);

        // Create EPs for local worker
        // Allocate address for all(local and remote) workers
        for (auto & iaddress: ucp_worker_address_book_) {
            iaddress = (ucp_address_t *)malloc(ucp_worker_address_len);
        }
        // Copy local worker address to address table
        for(int i = 0; i < socket_num_; i++){
            memcpy(ucp_worker_address_book_[rank_ * socket_num_ + i], ucp_worker_address_[i], ucp_worker_address_len);
        }
    
        // Using MPI to broadcast address from all ranks to all ranks(all broadcast)
        for (int iroot = 0; iroot < num_proc_; iroot++) {
            for( int i = 0; i < socket_num_; i++){
                MPI_Bcast(ucp_worker_address_book_[iroot * socket_num_ + i], ucp_worker_address_len, MPI_BYTE, iroot, comm_);
            }
        }

        // Create EPs on local worker to other workers(include itself)
        for(int socket = 0; socket < socket_num_; socket++){
            for(int i = 0; i < socket_num_ * num_proc_; i++){
                // Only need to set once
                if( socket == 0 ){
                    memset(&ucp_ep_params[i], 0, sizeof(ucp_ep_params[i]));
                    ucp_ep_params[i].field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
                    ucp_ep_params[i].address = ucp_worker_address_book_[i];
                }
                ucp_ep_create(ucp_worker_[socket], &ucp_ep_params[i], &ucp_endpoints_[socket][i]);
            }
        }

        // Allocate affinity list for all GPUs on all nodes
        for(int i = 0; i < num_proc_; i++){

            affinity_list_[i] = (gpu_id_t *)malloc(num_local_gpu_ * sizeof(*affinity_list_[i]));
            
        }
        
        // Assign each local T-GPU to the local L-socket
        for(int i = 0; i < num_local_gpu_; i++){

            // Find the affinity CPU set that current topo GPU is binding to
            hwloc_cudart_get_device_cpuset(topo_, GPU_list_[i], cpu_set);
            hwloc_obj_t affinity_socket = hwloc_get_next_obj_covering_cpuset_by_type(topo_, cpu_set, HWLOC_OBJ_PACKAGE, NULL);
            affinity_list_[rank_][i] = (gpu_id_t)(affinity_socket -> logical_index);

        }

        // Using MPI to broadcast GPU locality info to all other ranks
        for (int iroot = 0; iroot < num_proc_; iroot++) {

            MPI_Bcast(affinity_list_[iroot], num_local_gpu_ * sizeof(*affinity_list_[iroot]), MPI_BYTE, iroot, comm_);

        }

        hwloc_bitmap_free(ori_cpu_set);
        hwloc_bitmap_free(cpu_set);

    }

    // Dtor
    ~FasterGossipCommMulti(){
        //free(request_);
        for(int stage = 0; stage < num_proc_; stage++){

            delete GossipCommHandle_[stage];

        }
        

        // Release UCP EPs
        for(int socket = 0; socket < socket_num_; socket++){

            for (int irank = 0; irank < socket_num_ * num_proc_; irank++) {

                // Flush all operations associated with the EP and release the EP
                ucs_status_ptr_t ucs_status_ptr = ucp_ep_close_nb(ucp_endpoints_[socket][irank], UCP_EP_CLOSE_MODE_FLUSH);

                if(UCS_PTR_IS_ERR(ucs_status_ptr) || UCS_PTR_STATUS(ucs_status_ptr) == UCS_OK){
                    continue;
                }

                // While the releasing is not finished, progress the worker
                while(ucp_request_check_status(ucs_status_ptr) == UCS_INPROGRESS){
                    for(int j = 0; j < socket_num_; j++){
                        ucp_worker_progress( ucp_worker_[j] );
                    }
                }

                // Free the request
                ucp_request_free( ucs_status_ptr );

            }

        }

        // Wait for all ranks to release EPs before releasing any worker
        MPI_Barrier(comm_);

        // Release worker address
        for( int i = 0; i < socket_num_; i++){
            ucp_worker_release_address(ucp_worker_[i], ucp_worker_address_[i]);
        }
    
        // Release worker
        for( int i = 0; i < socket_num_; i++){
            ucp_worker_destroy(ucp_worker_[i]);
        }

        // Release UCP context
        for( int i = 0; i < socket_num_; i++){

            ucp_cleanup(ucp_context_[i]);

        }

        // Free address book
        for (auto & iaddress : ucp_worker_address_book_) {
            free(iaddress);
        }

        // Free HWLOC topology
        hwloc_topology_destroy(topo_);

        // Free GPU affinity list
        for(int i = 0; i < num_proc_; i++){

            free(affinity_list_[i]);
            
        }

    }

    // Initialize a communication
    void Initialize(const std::vector<data_t_ *>& src,
                    const std::vector<data_t_ *>& dst,
                    const std::vector<std::vector<size_t>>& send_table,
                    const std::vector<std::vector<size_t>>& recv_table){

        // Device restorer
        nv::CudaDeviceRestorer dev_restorer;

        // record user provide data
        src_ = src;
        dst_ = dst;
        send_table_ = send_table;
        recv_table_ = recv_table;

        // Calculate the size of Local buffers and Recv buffers, and allocate on each local GPU
        for(int i = 0; i < num_local_gpu_; i++){

            size_t max_size = 0;

            for (int j = 0; j < num_proc_; j++){

                if(j != rank_){

                    size_t accum_size = 0;

                    for(int k = 0; k < num_local_gpu_; k++){

                        accum_size += recv_table_[k][i + j * num_local_gpu_];

                    }

                    max_size = std::max(max_size, accum_size);

                }
            }

            // Allocate buffers on current topo GPU
            CUDA_CHECK( cudaSetDevice( GPU_list_[i] ) );
            CUDA_CHECK( cudaMalloc( &local_buffer_[i], sizeof(data_t_) * max_size ) );
            CUDA_CHECK( cudaMalloc( &recv_buffer_[i], sizeof(data_t_) * max_size) );
            
        }

        // Max buffer size required by gossip all2all on each GPU
        std::vector<size_t> max_temp_buf_size(num_local_gpu_, 0);

        // Initialize all gossip all2all object
        for(int stage = 0; stage < num_proc_; stage++){

            // for first stage, do all2all on local data
            if(stage == 0){

                // Extract the temp table for local all2all on this stage
                for(int i = 0; i < num_local_gpu_; i++){

                    for(int j = 0; j < num_local_gpu_; j++){

                        temp_table_[i][j] = recv_table_[j][rank_ * num_local_gpu_ + i];

                    }

                }

                // Extract the temp src and dst buffers for local all2all on this stage
                for(int i = 0; i < num_local_gpu_; i++){

                    size_t src_offset = 0;
                    size_t dst_offset = 0;
                    for(int j = 0; j < num_local_gpu_ * rank_; j++){

                        src_offset += send_table_[i][j];
                        dst_offset += recv_table_[i][j];

                    }
                    temp_src_[i] = src_[i] + src_offset;
                    temp_dst_[i] = dst_[i] + dst_offset;

                }
                
                // Initialize the local all2all
                std::vector<size_t> temp_buf_size = GossipCommHandle_[stage]->Initialize_no_malloc(temp_src_, temp_dst_, temp_table_);
                
                // Find the largest buffer size needed on each GPU
                for(int i = 0; i < num_local_gpu_; i++){

                    max_temp_buf_size[i] = std::max(temp_buf_size[i], max_temp_buf_size[i]);

                }

            }
            // for later stage, do all2all with data received from previous stage
            else{

                // previous stage src node
                int prev_src_node = (rank_ + num_proc_ - stage) % num_proc_;

                // Extract the temp table for local all2all on this stage
                for(int i = 0; i < num_local_gpu_; i++){

                    for(int j = 0; j < num_local_gpu_; j++){

                        temp_table_[i][j] = recv_table_[j][prev_src_node * num_local_gpu_ + i];

                    }

                }

                // Extract the temp dst buffers for local all2all on this stage
                for(int i = 0; i < num_local_gpu_; i++){

                    size_t dst_offset = 0;
                    for(int j = 0; j < num_local_gpu_ * prev_src_node; j++){

                        dst_offset += recv_table_[i][j];

                    }
                    temp_dst_[i] = dst_[i] + dst_offset;

                }

                std::vector<size_t> temp_buf_size;

                // Initialize the local all2all
                if(stage % 2 == 0){
                    temp_buf_size = GossipCommHandle_[stage]->Initialize_no_malloc(local_buffer_, temp_dst_, temp_table_);
                }
                else{
                    temp_buf_size = GossipCommHandle_[stage]->Initialize_no_malloc(recv_buffer_, temp_dst_, temp_table_);
                }

                // Find the largest buffer size needed on each GPU
                for(int i = 0; i < num_local_gpu_; i++){

                    max_temp_buf_size[i] = std::max(temp_buf_size[i], max_temp_buf_size[i]);

                }

            }

        }

        // Allocate max size temp buffers shared by all gossip all2all
        for(int i = 0; i < num_local_gpu_; i++){
            // Allocate temp buffers on each GPU
            CUDA_CHECK( cudaSetDevice( GPU_list_[i] ) );
            CUDA_CHECK( cudaMalloc( &temp_buf_[i], sizeof(data_t_) * max_temp_buf_size[i] ) );
        }

        // Set the allocated temp buffers to all gossip all2all
        for(int stage = 0; stage < num_proc_; stage++){

            GossipCommHandle_[stage]->set_buf(temp_buf_);

        }

        // Run exec() in advance to warm up all buffers used by UCX
        // For even nodes, 1 run is enough for warm up, for odd nodes, 2 runs is needed
        for(int i = 0; i < WARM_UP_ROUND; i++){
            exec();
        }
        
    }

    void exec(){

        // loop through all stages
        for(int stage = 0; stage < num_proc_; stage++){

            // We cuse 2 threads, one for UCX P2P, one for gossip all2all. In the same stage, these 2 operations
            // can be executed concurrently
            #pragma omp parallel default(none) shared(stage, num_proc_, rank_, num_local_gpu_, \
                                                      send_table_, affinity_list_, send_reqs_, ucp_endpoints_,\
                                                      socket_num_, src_, recv_table_, recv_reqs_, \
                                                      ucp_worker_, recv_buffer_,\
                                                      GossipCommHandle_) num_threads(2)
            {

                // Each thread grab its ID within this OpenMP thread team
                int thread_id = omp_get_thread_num();

                // Thread 0 do the gossip all2all
                if(thread_id == 0){

                    // do local all2all
                    // Execute the local all2all
                    GossipCommHandle_[stage]->execAsync();

                    // wait for local all2all to complete
                    GossipCommHandle_[stage]->sync();

                }
                // Thread 1 do the UCX P2P
                else{

                    // for all stage except last stage, send and receive data to/from other nodes
                    if(stage < num_proc_ -1){

                        // The dst and src rank of local node in this stage
                        int dst_rank = (rank_ + stage + 1) % num_proc_;
                        int src_rank = (rank_ + num_proc_ - stage - 1) % num_proc_;
                        // loop through all local GPUs to send GPU buffers to dst worker
                        for(int i = 0; i < num_local_gpu_; i++){
                    
                            size_t src_offset = 0;
                            size_t src_len = 0;
                            // Accumulate the offset within the src_buffer
                            for(int j = 0; j < num_local_gpu_ * dst_rank; j++){

                                src_offset += send_table_[i][j];

                            }
                            // Accumulate the amount of elements to send to the target node
                            for(int j = 0; j < num_local_gpu_; j++){

                                src_len += send_table_[i][j + num_local_gpu_ * dst_rank];

                            }
                    
                            //MPI_Isend(src_[i] + src_offset, sizeof(data_t_) * src_len, MPI_BYTE, dst_rank, i, comm_, request_ + i);

                            // Prepare the tag for tag-matching massage passing, the tag should identify the user tag, source worker of the tag and other info 
                            ucp_tag_t comm_tag = 0LLU;
                            // MSB 32-bit for original MPI TAG
                            comm_tag |= ((ucp_tag_t)i << 32);
                            // 16-32 bits are source rank
                            comm_tag |= ((ucp_tag_t)(rank_ & 0x0000FFFF) << 16);
                            // The 0-15 bits are source L-socket(worker)
                            comm_tag |= (((ucp_tag_t)(affinity_list_[rank_][i])) & 0x000000000000FFFF);

                            send_reqs_[i] = ucp_tag_send_nb(
                                ucp_endpoints_[affinity_list_[rank_][i]][dst_rank * socket_num_ + affinity_list_[dst_rank][i]], src_[i] + src_offset,
                                sizeof(data_t_) * src_len, ucp_dt_make_contig(sizeof(char)),
                                comm_tag, empty_send_callback_func
                            );

                            // If the returned request is not a valid pointer, that means that the operation already finished(failed or completed), the callback will not been 
                            // called in these situation and the returned request is not de-referencable thus no release needed. 
                            if(UCS_PTR_IS_ERR(send_reqs_[i]) || UCS_PTR_STATUS(send_reqs_[i]) == UCS_OK){
                                send_reqs_[i] = nullptr;
                            }

                        }

                        // loop through all local GPUs to receive GPU buffers from src worker
                        for(int i = 0; i < num_local_gpu_; i++){

                            size_t dst_len = 0;
                            // Accumulate the amount of elements to receive from the source node
                            for(int j = 0; j < num_local_gpu_; j++){

                                dst_len += recv_table_[j][i + src_rank * num_local_gpu_];

                            }

                            //MPI_Irecv(recv_buffer_[i], sizeof(data_t_) * dst_len, MPI_BYTE, src_rank, i, comm_, request_ + num_local_gpu_ +i);

                            // Prepare the tag for tag-matching massage passing, the tag should identify the user tag, source worker of the tag and other info 
                            ucp_tag_t comm_tag = 0LLU;
                            // MSB 32-bit for original MPI TAG
                            comm_tag |= ((ucp_tag_t)i << 32);
                            // 16-32 bits are source rank
                            comm_tag |= ((ucp_tag_t)(src_rank & 0x0000FFFF) << 16);
                            // The 0-15 bits are source L-socket(worker)
                            comm_tag |= (((ucp_tag_t)(affinity_list_[src_rank][i])) & 0x000000000000FFFF);


                            recv_reqs_[i] = ucp_tag_recv_nb(
                                ucp_worker_[affinity_list_[rank_][i]], recv_buffer_[i],
                                sizeof(data_t_) * dst_len, ucp_dt_make_contig(sizeof(char)),
                                comm_tag, (ucp_tag_t)-1, empty_recv_callback_func
                            );

                            // The same as send, but recv API never return UCS_OK, only UCS_ERR_xx or valid pointer can be returned
                            if(UCS_PTR_IS_ERR(recv_reqs_[i])){
                                recv_reqs_[i] = nullptr;
                            }

                        }

                    }

                    // for all stage except last stage, wait for UCX communication to finish
                    if(stage < num_proc_ -1){

                        // Wait for all send to finish
                        for(int i = 0; i < num_local_gpu_; i++){

                            // If the current operation is not completed yet, progress it
                            while( send_reqs_[i] != nullptr && ucp_request_check_status(send_reqs_[i]) ==  UCS_INPROGRESS){
                                for(int j = 0; j < socket_num_; j++){
                                    ucp_worker_progress( ucp_worker_[j] );
                                }
                            }

                        }

                        // Wait for all receive to finish
                        for(int i = 0; i < num_local_gpu_; i++){

                            // If the current operation is not completed yet, progress it
                            while( recv_reqs_[i] != nullptr && ucp_request_check_status(recv_reqs_[i]) ==  UCS_INPROGRESS){
                                for(int j = 0; j < socket_num_; j++){
                                    ucp_worker_progress( ucp_worker_[j] );
                                }
                            }

                        }

                        // Da-allocate UCP request before going to next round
                        for(int i = 0; i < num_local_gpu_; i++){

                            if(send_reqs_[i] != nullptr){
                                ucp_request_free( send_reqs_[i] );
                                send_reqs_[i] = nullptr;
                            }

                            if(recv_reqs_[i] != nullptr){
                                ucp_request_free( recv_reqs_[i] );
                                recv_reqs_[i] = nullptr;
                            }

                        }
                        //MPI_Waitall(2 * num_local_gpu_, request_, MPI_STATUSES_IGNORE);
                    }

                }

            }

            // Swap recv_buffer and local_buffer pointer. If there is odd nodes, do not swap in the last stage
            if(num_proc_ % 2 != 0 && stage == num_proc_ - 1){
                continue;
            }

            recv_buffer_.swap(local_buffer_);

        }// stage loop

    }

    void reset(){

        // Device restorer
        nv::CudaDeviceRestorer dev_restorer;

        // Free local_buffer and recv_buffer, ready for next multi-node all2all
        for(int i = 0; i < num_local_gpu_; i++){
            // Free temp buffers on each GPU
            CUDA_CHECK( cudaSetDevice( GPU_list_[i] ) );
            CUDA_CHECK( cudaFree( local_buffer_[i] ) );
            CUDA_CHECK( cudaFree( recv_buffer_[i] ) );
        }

        // Free gossip all2all temp buffers
        for(int i = 0; i < num_local_gpu_; i++){

            CUDA_CHECK( cudaSetDevice( GPU_list_[i] ) );
            CUDA_CHECK( cudaFree( temp_buf_[i] ) );

        }

    }

private:

    // GPU list
    std::vector<gpu_id_t> GPU_list_;
    // GPU count
    gpu_id_t num_local_gpu_;
    gpu_id_t num_total_gpu_;
    // MPI-related resource
    int rank_;
    int num_proc_;
    MPI_Comm comm_;
    //MPI_Request * request_;
    // Local gossip all2all library
    std::vector<FasterGossipComm *> GossipCommHandle_;
    // Temp local GPU buffers for remote data
    std::vector<data_t_ *> local_buffer_;
    std::vector<data_t_ *> recv_buffer_;
    // Temp local GPU buffers for local all2all
    std::vector<data_t_ *> temp_buf_;
    // Buffers and tables provided by users
    std::vector<data_t_ *> src_;
    std::vector<data_t_ *> dst_;
    std::vector<std::vector<size_t>> send_table_;
    std::vector<std::vector<size_t>> recv_table_;
    // Temp table for local all2all
    std::vector<std::vector<size_t>> temp_table_;
    // Temp src and dst pinter vector for local all2all
    std::vector<data_t_ *> temp_src_;
    std::vector<data_t_ *> temp_dst_;
    // Socket count
    int socket_num_;
    // UCP variable: UCP context, UCP worker, UCP address, UCP EP and UCP request
    std::vector<ucp_context_h> ucp_context_;
    std::vector<ucp_worker_h>  ucp_worker_;
    std::vector<ucp_address_t *> ucp_worker_address_;
    std::vector<ucp_address_t *> ucp_worker_address_book_;
    std::vector<std::vector<ucp_ep_h>> ucp_endpoints_;
    std::vector<ucs_status_ptr_t> send_reqs_;
    std::vector<ucs_status_ptr_t> recv_reqs_;
    // HWLOC variable: topo
    hwloc_topology_t topo_;
    // The buffers that record the locality of each GPU in GPU list on each nodes
    std::vector<gpu_id_t *> affinity_list_;

}; // class

} // namespace


from collections import defaultdict
from itertools import chain
import threading
from time import sleep
class NcclComm:
    def __init__(self) -> None:
        self.send_dict = defaultdict(dict)
        self.recv_dict = defaultdict(dict)
        self.count = 0

    def all2all(self, num_gpus):
        for recv_id in range(num_gpus):
            for send_id in range(num_gpus):
                send_tensor, send_count = self.send_dict[send_id][recv_id]
                recv_tensor, recv_count = self.recv_dict[recv_id][send_id]
                if send_count != recv_count:
                    raise RuntimeError('nccl comm buffer size not match.')
                recv_tensor[:recv_count] = send_tensor[:send_count]

    def allreduce(self, num_gpus):
        buffer = [self.send_dict[gpu_id]['allreduce'][0] for gpu_id in range(num_gpus)]
        count = [self.send_dict[gpu_id]['allreduce'][1] for gpu_id in range(num_gpus)]
        assert sum(count) == count[0] * num_gpus

        allreduce_buffer = [0. for _ in range(count[0])]
        for gpu_id in range(num_gpus):
            for i in range(count[0]):
                allreduce_buffer[i] += buffer[gpu_id][i]
        for b in buffer:
            for i in range(count[0]):
                b[i] = allreduce_buffer[i]
                
    def clear(self):
        self.send_dict.clear()
        self.recv_dict.clear()


global_nccl_comm = NcclComm()
barrier = threading.Barrier(2)

class nccl_communication:
    def __init__(self, typ, gpu_id, num_gpus) -> None:
        self.typ = typ
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if barrier.parties != self.num_gpus:
            raise RuntimeError("please change num of barrier")
        barrier.wait()
        barrier.reset()
        
        if self.gpu_id == 0:
            if self.typ == 'all2all':
                global_nccl_comm.all2all(self.num_gpus)
            elif self.typ == 'allreduce':
                global_nccl_comm.allreduce(self.num_gpus)
            else:
                raise RuntimeError("not supported communication type {}".format(self.typ))

            global_nccl_comm.clear()

        barrier.wait()
        barrier.reset()
        

def nccl_send(send_tensor, send_count, peer, comm):
    global_nccl_comm.send_dict[comm.gpu_id][peer] = send_tensor, send_count

def nccl_recv(recv_tensor, recv_count, peer, comm):
    global_nccl_comm.recv_dict[comm.gpu_id][peer] = recv_tensor, recv_count


def nccl_allreduce(send_recv_buffer, count, comm):
    global_nccl_comm.send_dict[comm.gpu_id]['allreduce'] = send_recv_buffer, count
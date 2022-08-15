"""
 Copyright (c) 2022, NVIDIA CORPORATION.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# We don't want the whole process to quit bacause of the import failure when
# we don't use horovod to do communication.
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


class CommToolBase(object):
    """Abstract base class for different communication tools.

    This class assumes that a process can control multiple GPUs and there can be multiple
    processes on a node(a physical machine).

    A `context`(in the comments below) corresponds to a unique GPU, a typical example is
    the context of tensorflow's mirrored strategy.

    There should only be one instance of this class(in fact, its subclass) per process.
    """

    def rank(self):
        """Returns the global id for the current `process`.

        Similar to horovod, `rank` represents the global id of the current process. The
        difference is that `rank` is not equal to the global id of GPU since we don't
        assume that there is only one GPU in a process.

        `rank` belongs to [0, num_ranks-1].
        """
        raise NotImplementedError("rank() is not implemented")

    def num_ranks(self):
        """Returns how many processes are running in total.

        This includes all process on different nodes.
        """
        raise NotImplementedError("num_ranks() is not implemented")

    def local_rank(self):
        """Returns the local id of the current process in the current node.

        Similar to `rank`, `local rank` is not equal to the GPU id in current node since
        a process may control multiple GPUs.
        """
        raise NotImplementedError("local_rank() is not implemented")

    def id_in_rank(self):
        """Returns the local id of current `context` in current rank.

        `id_in_rank` belongs to [0, num_gpu_per_rank-1].
        """
        raise NotImplementedError("id_in_rank() is not implemented")

    def num_gpu_per_rank(self):
        """Returns how many GPUs a process controls."""
        raise NotImplementedError("num_gpu_per_rank() is not implemented")

    def global_gpu_id(self):
        """Returns the global id of GPU(in current context) among all participating GPUs.

        `global_gpu_id` belongs to [0, num_gpus-1].

        `global_gpu_id` should be equal to `rank * num_gpu_per_rank + id_in_rank`.
        """
        raise NotImplementedError("global_gpu_id() is not implemented")

    def num_gpus(self):
        """Returns how many GPUs are running in total.

        `num_gpus` should be equal to `num_ranks * num_gpu_per_rank`.
        """
        raise NotImplementedError("num_gpus() is not implemented")

    def alltoall(self, tensor, splits):
        """Performs an alltoall operation.

        Args:
          tensor: A tensorflow tensor to be sent.
          splits: A tensorflow tensor representing how much data should be sent to each GPU.
        """
        raise NotImplementedError("alltoall() is not implemented")

    def allreduce(self, tensor, op):
        """Performs an allreduce operation.

        Args:
          tensor: A tensorflow tensor to be sent.
          op    : A python string representing the reduce type.
        """
        raise NotImplementedError("allreduce() is not implemented")

    def allgather(self, tensor):
        """Performs an allgather operation.

        Args:
          tensor: A tensorflow tensor to be sent.
        """
        raise NotImplementedError("allgather() is not implemented")


class HorovodTool(CommToolBase):
    def rank(self):
        return hvd.rank()

    def num_ranks(self):
        return hvd.size()

    def local_rank(self):
        return hvd.local_rank()

    def id_in_rank(self):
        return 0

    def num_gpu_per_rank(self):
        return 1

    def global_gpu_id(self):
        return hvd.rank()

    def num_gpus(self):
        return hvd.size()

    def alltoall(self, tensor, splits):
        return hvd.alltoall(tensor, splits)

    def allreduce(self, tensor, op):
        # TODO: Add more op options
        if op == "sum":
            op = hvd.Sum
        elif op == "average":
            op = hvd.Average
        return hvd.allreduce(tensor, op=op)

    def allgather(self, tensor):
        return hvd.allgather(tensor)


# The global communication tool instance, it should only be set by `set_comm_tool` method.
_COMM_TOOL = None


def set_comm_tool(tool):
    """Set the communication tool.

    Note that the communication tool cannot be set more than once.
    """
    global _COMM_TOOL

    if _COMM_TOOL is not None:
        raise RuntimeError("[SOK INFO] Communication tool cannot be set more than once")

    # TODO: Add more communication tools
    if tool == "horovod":
        _COMM_TOOL = HorovodTool()


def check_comm_tool():
    if _COMM_TOOL is None:
        raise RuntimeError(
            "[SOK INFO] Communication tool is not set. Did you forget to call sok.init()?"
        )


def rank():
    check_comm_tool()
    return _COMM_TOOL.rank()


def num_ranks():
    check_comm_tool()
    return _COMM_TOOL.num_ranks()


def local_rank():
    check_comm_tool()
    return _COMM_TOOL.local_rank()


def id_in_rank():
    check_comm_tool()
    return _COMM_TOOL.id_in_rank()


def num_gpu_per_rank():
    check_comm_tool()
    return _COMM_TOOL.num_gpu_per_rank()


def global_gpu_id():
    check_comm_tool()
    return _COMM_TOOL.global_gpu_id()


def num_gpus():
    check_comm_tool()
    return _COMM_TOOL.num_gpus()


def alltoall(*args, **kwargs):
    check_comm_tool()
    return _COMM_TOOL.alltoall(*args, **kwargs)


def allreduce(*args, **kwargs):
    check_comm_tool()
    return _COMM_TOOL.allreduce(*args, **kwargs)


def allgather(*args, **kwargs):
    check_comm_tool()
    return _COMM_TOOL.allgather(*args, **kwargs)

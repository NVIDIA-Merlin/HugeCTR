# On DGX-1V, there are four Infinibind devices
export IBDEVICES="-device=/dev/infiniband/rdma_cm --device=/dev/infiniband/ucm3 --device=/dev/infiniband/ucm2 --device=/dev/infiniband/ucm1 --device=/dev/infiniband/ucm0 --device=/dev/infiniband/uverbs3 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm3 --device=/dev/infiniband/issm2 --device=/dev/infiniband/issm1 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad3 --device=/dev/infiniband/umad2 --device=/dev/infiniband/umad1 --device=/dev/infiniband/umad0"


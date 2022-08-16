# working dir, dataset dir, bin path, config path
WORK_DIR="/hugectr"
TEST_CMD="python3 /hugectr/samples/dcn/dcn_2node_8gpu.py"
DATASET="/dataset"
VOL_DATASET="/dataset"
IMAGENAME="nvcr.io/nvidia/merlin/merlin-hugectr:22.06"
HOSTS="node1,node2"
GPUIDS="0,1,2,3,4,5,6,7"
CONTNAME="hugectr_test"
PORT=9999
NIC_SPECIFIC=''

ARGS=$(getopt -o 'w:t:d:v:i:o:g:c:p:s:h' -l 'work_dir:,test_cmd:,dataset:,vol_datase:,image:,hosts:,gpuids:,contname:,port:,specific_nic:,help' -- "$@")
if [ $? != 0 ] ; then echo "Parse error! Terminating..." >&2 ; exit 1 ; fi
eval set -- "$ARGS"

while true ; do
     case "$1" in
          -w|--work_dir) WORK_DIR="$2" ; shift 2 ;;
          -t|--test_cmd) TEST_CMD="$2" ; shift 2 ;;
          -d|--dataset) DATASET="$2" ; shift 2 ;;
          -v|--vol_dataset) VOL_DATASET="$2" ; shift 2 ;;
          -i|--image) IMAGENAME="$2" ; shift 2 ;;
          -o|--hosts) HOSTS="$2" ; shift 2 ;;
          -g|--gpuids) GPUIDS="$2" ; shift 2 ;;
          -c|--contname) CONTNAME="$2" ; shift 2 ;;
          -p|--port) PORT="$2" ; shift 2 ;;
          -s|--specific_nic) NIC_SPECIFIC="$2" ; shift 2 ;;
          -h|--help)  
             echo "args:"
             echo "args name: -w|--work_dir          :  Docker container working path,recommended is HugeCTR code repository path,must use absolute path,default=/hugectr"
             echo "args name: -t|--test_cmd          :  How to run python script,default=python3 /hugectr/samples/dcn/dcn_2node_8gpu.py"
             echo "args name: -d|--dataset           :  Real dataset path,default=/dataset"
             echo "args name: -v|--vol_dataset       :  Dataset path as shown inside your docker container as a mapping from DATASET. HugeCTR only sees this path.,default=/dataset"
             echo "args name: -i|--image             :  Name of your Docker image.,default=nvcr.io/nvidia/merlin/merlin-hugectr:22.06"
             echo "args name: -o|--hosts             :  The IPs of multinode,IPs are separated by commas,default=node1,node2"
             echo "args name: -g|--gpuids            :  GPU IDs map into container,default=0,1,2,3,4,5,6,7"
             echo "args name: -c|--contname          :  Docker container name,default=hugectr_test"
             echo "args name: -p|--port              :  Select a net port use in container,default=9999"
             echo "args name: -s|--specific_nic      :  in some physical , NCCL in docker container can find NIC correct ,specific nic use rule of NCCL env NCCL_IB_HCA ,default=empty str"
             echo "args name: -h|--help              :  Show args info"
             exit 0
          ;;
          --) shift ; break ;;
          *) echo "Internal error!" ; exit 1 ;;
     esac
done


# This mpirun is used to do some hostfile and keys generatoin and launch container in host
JOBID="hugectr-test-$(date +%s)"
MPIRUN="mpirun --allow-run-as-root"

IFS=',' read -r -a hosts <<< "$HOSTS"

# 1. Pull docker image  
$MPIRUN -H $HOSTS docker pull $IMAGENAME

echo "FINISH IMAGE PULLING"

# 2. Prepare docker run config and build container
DOCKERRUN_ARGS=(
                 --rm
                 --init
                 --runtime=nvidia
                 --network=host
                 --uts=host
                 --ipc=host
                 --ulimit stack=67108864
                 --shm-size=1g
                 --ulimit memlock=-1
                 --security-opt seccomp=unconfined
                 --cap-add=SYS_NICE
                 $IBDEVICES
                 --name $CONTNAME
               )


VOLS="-v ${WORK_DIR}:/hugectr -v $DATASET:$VOL_DATASET"
VARS=(
       -e "NVIDIA_VISIBLE_DEVICES=$GPUIDS"
       -e "OMPI_MCA_plm_rsh_agent=ssh"
     )

$MPIRUN -H $HOSTS docker run -d "${DOCKERRUN_ARGS[@]}" $VOLS -w /hugectr "${VARS[@]}" ${IMAGENAME} bash -c 'sleep infinity'; rv=$?
[[ $rv -ne 0 ]] && echo "ERR: Container launch failed." && exit $rv
sleep 10

# 3. Generate ssh key in docker container
RUN_SCRIPT_STR="#!/bin/bash\n"
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"mkdir -p ~/.ssh && ssh-keygen -t rsa -f ~/.ssh/id_rsa -C $CONTNAME\n"
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\n"
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"cat>~/.ssh/config<<EOF\n"
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"StrictHostKeyChecking no\n"
for single_host in ${hosts[*]}
do
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"Host=${single_host}\n"
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"Port=${PORT}\n"
done
RUN_SCRIPT_STR=$RUN_SCRIPT_STR"EOF\n"
$MPIRUN -H $HOSTS docker exec  $CONTNAME bash -c "echo -e \"$RUN_SCRIPT_STR\">~/start_sshd.sh && chmod 777 ~/start_sshd.sh && bash ~/start_sshd.sh"
#
## 4.Start sshd service in docker container
$MPIRUN -H $HOSTS docker exec  $CONTNAME bash -c "sed -i \"s/\#Port 22/Port ${PORT}/\" \`grep \"\#Port 22\" -rl /etc/ssh/sshd_config\`";
$MPIRUN -H $HOSTS docker exec  $CONTNAME bash -c "sed -i \"s/\#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/\" \`grep \"\#ListenAddress 0.0.0.0\" -rl /etc/ssh/sshd_config\`";
$MPIRUN -H $HOSTS docker exec  $CONTNAME bash -c "/etc/init.d/ssh start"; 
#
## 5. Exchange public key
ssh_pub_key=`$MPIRUN -H $HOSTS docker exec $CONTNAME bash -c "cat ~/.ssh/authorized_keys"`;
$MPIRUN -H $HOSTS docker exec $CONTNAME bash -c "echo \"${ssh_pub_key}\">~/.ssh/authorized_keys";


# 6. Start triaining
if [ $NIC_SPECIFIC ];then
    NIC_SPECIFIC="NCCL_IB_HCA="$NIC_SPECIFIC
fi

MPI_CMD_ARGS=(
              -H $HOSTS 
              -np $((${#hosts[@]})) 
              -x NCCL_DEBUG=INFO 
              -x LD_LIBRARY_PATH 
              -x PATH
              -x PYTHONPATH
              -x ${NIC_SPECIFIC}
)

docker exec $CONTNAME mpirun --allow-run-as-root --bind-to none "${MPI_CMD_ARGS[@]}" ${TEST_CMD} 2>&1|tee $JOBID.log 

#echo "FINISH TRAINING"
$MPIRUN -H $HOSTS docker stop $CONTNAME
echo "FINISH ENV CLEAN-UP"

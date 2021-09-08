# working dir, dataset dir, bin path, config path
WORK_DIR="../../build/bin"
TEST_CMD="python3 ../../samples/dcn/dcn_2node_8gpu.py"
DATASET="/dataset"
VOL_DATASET="/dataset"
IMAGENAME="hugectr:devel_train"

export HOSTS="node1,node2"

# Assign GPU index to be used for computing;
export GPUIDS="0,1,2,3,4,5,6,7"
# Docker image name and container name to be started
CONTNAME="hugectr_test"

# Setting for temporary directory for generating hostfile and keys
DIRNAME="/dev/shm/mpi_tmp"
JOBID="hugectr-test-$(date +%s)"

# This mpirun is used to do some hostfile and keys generatoin and launch container in host
MPIRUN="mpirun --allow-run-as-root"


# ENV CONFIGURATION

# 1. Prepare run dir on all nodes
$MPIRUN -H $HOSTS mkdir -p ${DIRNAME}/${JOBID}
$MPIRUN -H $HOSTS chmod 700 ${DIRNAME}/${JOBID}/

echo "FINISH RUN DIR PREPARATION"

# 2. Distribute ssh helper script that transfers an ssh into a compute node into the running container on that node
$MPIRUN -H $HOSTS -x DIRNAME=${DIRNAME} -x JOBID=${JOBID} -x CONTNAME=$CONTNAME bash -c 'tee ${DIRNAME}/${JOBID}/sshentry.sh >/dev/null <<EOF
#!/bin/bash
echo "::sshentry: entered \$(hostname)"
echo "::sshentry: running \$SSH_ORIGINAL_COMMAND"
exec docker exec $CONTNAME /bin/bash -c "\$SSH_ORIGINAL_COMMAND"
EOF'
$MPIRUN -H $HOSTS chmod 755 ${DIRNAME}/${JOBID}/sshentry.sh

echo "FINISH SSH HELPER SCRIPT DISTRIBUTION"

# 3. Make keys and distribute
ssh-keygen -t rsa -b 2048 -P "" -f "${DIRNAME}/${JOBID}/sshkey.rsa" -C "mpi_${JOBID}"  &>/dev/null
AUTHORIZED_KEY="command=\"${DIRNAME}/${JOBID}/sshentry.sh\",no-port-forwarding,no-agent-forwarding,no-X11-forwarding $(cat ${DIRNAME}/${JOBID}/sshkey.rsa.pub)"

$MPIRUN -H $HOSTS sed -i '$a '"${AUTHORIZED_KEY}"'' ${HOME}/.ssh/authorized_keys

echo "FINISH SSH KEY DISTRIBUTION"

# 4. Get all Host name and Create mpi hostlist
IFS=',' read -r -a hosts <<< "$HOSTS"

for hostn in ${hosts[@]}; do
   echo "$hostn slots=1" >> ${DIRNAME}/${JOBID}/mpi_hosts
done
cat ${DIRNAME}/${JOBID}/mpi_hosts

echo "FINISH MPI HOSTLIST CREATION"

# 5. Create mpi config file
cat > ${DIRNAME}/${JOBID}/mca_params.conf <<EOF
plm_rsh_agent = /usr/bin/ssh
plm_rsh_args = -i ${DIRNAME}/${JOBID}/sshkey.rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -l ${USER} -p 22
orte_default_hostfile = ${DIRNAME}/${JOBID}/mpi_hosts
btl_openib_connect_udcm_max_retry = 5000
btl_openib_warn_default_gid_prefix = 0
mpi_warn_on_fork = 0
btl_tcp_if_exclude = lo,docker,cni,veth,flannel
EOF

echo "FINISH MPI CONFIG FILE CREATION"

# START TRAINGING

#$MPIRUN -H $HOSTS docker pull $IMAGENAME

echo "FINISH IMAGE PULLING"

DOCKERRUN_ARGS=(
                 --rm
                 --init
                 --runtime=nvidia
                 --net=host
                 --uts=host
                 --ipc=host
                 --ulimit stack=67108864
                 --shm-size=1g
                 --ulimit memlock=-1
                 --security-opt seccomp=unconfined
                 --cap-add=SYS_ADMIN
                 $IBDEVICES
                 --name $CONTNAME
               )
VOLS="-v ${WORK_DIR}:/hugectr -v $DATASET:$VOL_DATASET"

VARS=(
       -e "NVIDIA_VISIBLE_DEVICES=$GPUIDS"
       -e "OMPI_MCA_mca_base_param_files=${DIRNAME}/${JOBID}/mca_params.conf"
     )

$MPIRUN -H $HOSTS docker run -d "${DOCKERRUN_ARGS[@]}" $VOLS -w /hugectr "${VARS[@]}" ${IMAGENAME} bash -c 'sleep infinity'; rv=$?

[[ $rv -ne 0 ]] && echo "ERR: Container launch failed." && exit $rv

sleep 10

echo "FINISH CONTAINER CREATION"

# You can adjust the mpirun args and adjust the running command, args
docker exec $CONTNAME mpirun --allow-run-as-root --bind-to none -np $((${#hosts[@]})) -x NCCL_DEBUG=INFO ${TEST_CMD} 2>&1 |tee $JOBID.log

echo "FINISH TRAINING"

# CLEAN ENV

$MPIRUN -H $HOSTS docker stop $CONTNAME
$MPIRUN -H $HOSTS rm -fr ${DIRNAME}/${JOBID}
$MPIRUN -H $HOSTS sed -i '/'"${CONTNAME}"'/d' ${HOME}/.ssh/authorized_keys

echo "FINISH ENV CLEAN-UP"

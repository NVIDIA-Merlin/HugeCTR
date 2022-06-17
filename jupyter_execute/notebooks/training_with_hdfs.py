#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # HugeCTR training with HDFS example

# ## Overview

# In version v3.4, we introduced the support for HDFS. Users can now move their data and model files from HDFS to local filesystem through our API to do HugeCTR training. And after training, users can choose to dump the trained parameters and optimizer states into HDFS. In this example notebook, we are going to demonstrate the end to end procedure of training with HDFS.

# ## Get HugeCTR from NGC
# The HugeCTR Python module is preinstalled in the 22.06 and later [Merlin Training Container](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-training): `nvcr.io/nvidia/merlin/merlin-training:22.06`.
# 
# You can check the existence of required libraries by running the following Python code after launching the container.
# 
# ```bash
# $ python3 -c "import hugectr"
# ```
# 
# > If you prefer to build HugeCTR from the source code instead of using the NGC container, refer to the 
# > [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development)
# > documentation.

# ## Hadoop Installation and Configuration
# 
# ### Download and Install Hadoop
# 
# 1. Download a JDK:
# 
#    ```bash
#    wget https://download.java.net/java/GA/jdk16.0.2/d4a915d82b4c4fbb9bde534da945d746/7/GPL/openjdk-16.0.2_linux-x64_bin.tar.gz
#    tar -zxvf openjdk-16.0.2_linux-x64_bin.tar.gz
#    mv jdk-16.0.2 /usr/local
#    ```
# 
# 2. Set Java environmental variables:
# 
#    ```bash
#    export JAVA_HOME=/usr/local/jdk-16.0.2
#    export JRE_HOME=${JAVA_HOME}/jre
#    export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
#    export PATH=.:${JAVA_HOME}/bin:$PATH
#    ```
# 
# 3. Download and install Hadoop:
# 
#    ```bash
#    wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz
#    tar -zxvf hadoop-3.3.1.tar.gz
#    mv hadoop-3.3.1 /usr/local
#    ```

# ### Hadoop configuration
# 
# Set Hadoop environment variables:
# 
# ```bash
# export HDFS_NAMENODE_USER=root
# export HDFS_DATANODE_USER=root
# export HDFS_SECONDARYNAMENODE_USER=root
# export YARN_RESOURCEMANAGER_USER=root
# export YARN_NODEMANAGER_USER=root
# echo ‘export JAVA_HOME=/usr/local/jdk-16.0.2’ >> /usr/local/hadoop/etc/hadoop/hadoop-env.sh
# ```
# 
# `core-site.xml` config:
# 
# ```xml
# <property>
#     <name>fs.defaultFS</name>
#     <value>hdfs://namenode:9000</value>
# </property>
# <property>
#     <name>hadoop.tmp.dir</name>
#     <value>/usr/local/hadoop/tmp</value>
# </property>
# ```
# 
# `hdfs-site.xml` for name node:
# 
# ```xml
# <property>
#      <name>dfs.replication</name>
#      <value>4</value>
# </property>
# <property>
#      <name>dfs.namenode.name.dir</name>
#      <value>file:/usr/local/hadoop/hdfs/name</value>
# </property>
# <property>
#      <name>dfs.client.block.write.replace-datanode-on-failure.enable</name>         
#      <value>true</value>
# </property>
# <property>
#     <name>dfs.client.block.write.replace-datanode-on-failure.policy</name>
#     <value>NEVER</value>
# </property>
# ```
# 
# `hdfs-site.xml` for data node:
# 
# ```xml
# <property>
#      <name>dfs.replication</name>
#      <value>4</value>
# </property>
# <property>
#      <name>dfs.datanode.data.dir</name>
#      <value>file:/usr/local/hadoop/hdfs/data</value>
# </property>
# <property>
#      <name>dfs.client.block.write.replace-datanode-on-failure.enable</name>         
#      <value>true</value>
# </property>
# <property>
#     <name>dfs.client.block.write.replace-datanode-on-failure.policy</name>
#     <value>NEVER</value>
# </property>
# ```
# 
# `workers` for all node:
# 
# ```bash
# worker1
# worker2
# worker3
# worker4
# ```

# ### Start HDFS
# 
# 1. Enable ssh connection:
# 
#    ```bash
#    ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
#    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
#    /etc/init.d/ssh start
#    ```
# 
# 2. Format the NameNode:
# 
#    ```bash
#    /usr/local/hadoop/bin/hdfs namenode -format
#    ```
# 
# 3. Format the DataNodes:
# 
#    ```bash
#    /usr/local/hadoop/bin/hdfs datanode -format
#    ```
# 
# 4. Start HDFS from the NameNode:
# 
#    ```bash
#    /usr/local/hadoop/sbin/start-dfs.sh
#    ```

# ## Wide and Deep Model
# 
# In the Docker container, `nvcr.io/nvidia/merlin/merlin-training:22.06`, 
# make sure that you installed Hadoop and set the proper environment variables as instructed in the preceding sections.
# 
# If you chose to compile HugeCTR, make sure you that you set `DENABLE_HDFS` to `ON`.
# 
# * Run `export CLASSPATH=$(hadoop classpath --glob)` first to link the required JAR file.
# * Make sure that we have the model files your Hadoop cluster and provide the correct links to the model files.
# 
# Now you can run the following sample.

# In[1]:


get_ipython().run_cell_magic('writefile', 'train_with_hdfs.py', 'import hugectr\nfrom mpi4py import MPI\nfrom hugectr.data import DataSource, DataSourceParams\n\ndata_source_params = DataSourceParams(\n    use_hdfs = True, #whether use HDFS to save model files\n    namenode = \'localhost\', #HDFS namenode IP\n    port = 9000, #HDFS port\n)\n\nsolver = hugectr.CreateSolver(max_eval_batches = 1280,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              repeat_dataset = True,\n                              data_source_params = data_source_params)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                                  source = [\'./wdl_norm/file_list.txt\'],\n                                  eval_source = \'./wdl_norm/file_list_test.txt\',\n                                  check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,\n                                    update_type = hugectr.Update_t.Global,\n                                    beta1 = 0.9,\n                                    beta2 = 0.999,\n                                    epsilon = 0.0000001)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        # the total number of slots should be equal to data_generator_params.num_slot\n                        [hugectr.DataReaderSparseParam("wide_data", 2, True, 1),\n                        hugectr.DataReaderSparseParam("deep_data", 1, True, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 69,\n                            embedding_vec_size = 1,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "wide_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 1074,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "deep_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"],\n                            top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,\n                            bottom_names = ["fc3", "reshape2"],\n                            top_names = ["add1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["add1", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\n\nmodel.load_dense_weights(\'/model/wdl/_dense_1000.model\')\nmodel.load_dense_optimizer_states(\'/model/wdl/_opt_dense_1000.model\')\nmodel.load_sparse_weights([\'/model/wdl/0_sparse_1000.model\', \'/model/wdl/1_sparse_1000.model\'])\nmodel.load_sparse_optimizer_states([\'/model/wdl/0_opt_sparse_1000.model\', \'/model/wdl/1_opt_sparse_1000.model\'])\n\nmodel.fit(max_iter = 1020, display = 200, eval_interval = 500, snapshot = 1000, snapshot_prefix = "/model/wdl/")\n')


# In[2]:


get_ipython().system('python train_with_hdfs.py')


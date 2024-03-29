# Environment Variables #
There are two kinds of environment variables in SparseOperationKit (SOK), which are compile variable and runtime variable, respectively.

## Compile Variables ##
These variables are available during building SOK.

### SOK_COMPILE_GPU_SM ###
This variable is used to configure the compute capability of the target GPU devices. By default, it equals to `70;75;80`. If you want to specify other values for this env var, please use semicolon to separate multiple SM values. For example, 
```shell
$ export SOK_COMPILE_GPU_SM="60;65;70;75;80"
```

### SOK_COMPILE_ASYNC ###
This variable is used to configure whether to enable dedicated CUDA stream for SOK's Ops. By default, it is enabled. 
#### values accepted ####
"0", "OFF", "Off", "off": these values indicates SOK will use the same CUDA stream as that of TF used. For example,
```shell
$ export SOK_COMPILE_ASYNC="OFF"
```

### SOK_COMPILE_USE_NVTX ###
This variable is used to indicate whether NVTX marks are enabled in SOK's Ops. By default, it is disabled.
#### values accepted ####
"1", "ON", "On", "on": these values indicates SOK will enable nvtx marks in its Ops. For example,
```shell
$ export SOK_COMPILE_USE_NVTX=1
```

### SOK_COMPILE_BUILD_TYPE ###
This variable is used to indicate whether the building is in `Release` or `Debug` mode. By default, it will be built at `Release` mode. If switched to `Debug` mode, there will be some debugging functionalities enabled, for example, assertion.
#### values accepted ####
"DEBUG", "Debug", "debug": these values indicates SOK will be built in `Debug` mode. For example,
```shell
$ export SOK_COMPILE_BUILD_TYPE="DEBUG"
```

### SOK_COMPILE_WITH_DEEPREC ###
This variable is used to indicate whether the building is with DEEPREC. By default, it will be built at `OFF` mode. If switched to `ON` mode, that means need to build SOK with DEEPREC, for example, assertion.
#### values accepted ####
"1", "ON", "On", "on": these values indicates SOK will compile with DEEPREC. For example,
```shell
$ export ENABLE_DEEPREC="ON"
```



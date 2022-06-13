# How to build and test:

- Get HCTR code

```shell
git clone https://gitlab-master.nvidia.com/dl/hugectr/hugectr.git
git checkout jamesrong/core-dev
```

- Start Docker container

```shell
docker run --privileged --gpus all -it --rm -v $(pwd):$(pwd) gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:devel_train
```

- Build and run Unittests

```shell
cd hugectr/HugeCTR/core/tf_impl/
bash ./build_and_test.sh
```
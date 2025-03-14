# Build HugeCTR Docker Containers

From 25.03, we provide a Dockerfile for building a base image in case you need a container to develop. You can build the develop image with command:
```sh
   docker build -t hugectr:devel -f tools/dockerfiles/Dockerfile.base .
```

# Documentation

This folder contains the scripts necessary to build the documentation for Hierarchical Parameter Server (HPS).
You can view the generated documents [here](https://nvidia-merlin.github.io/HugeCTR/hierarchical_parameter_server/master/hps_user_guide.html).

# Contributing to Docs

Follow the instructions below to be able to build the docs.

## Steps to follow:

1. To build the docs, create a developer environment for HugeCTR.

Use the following command to clone the HugeCTR repository:

`git clone https://github.com/NVIDIA/HugeCTR`

Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:

```shell
docker run --runtime=nvidia --rm -it --net=host --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-tensorflow:22.08
```

2. Install required documentation tools and extensions:

```shell
cd /hugectr/hierarchical_parameter_server/docs
pip install -r requirements-doc.txt
```

3. Build the documentation:

`make clean html`

The preceding command runs Sphinx in your shell and outputs to build/html/index.html.

View docs web page by opening HTML in browser:
First navigate to /build/html/ folder, i.e., cd build/html and then run the following command:

`python -m http.server 8000`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://localhost:8000`

Now you can check if your docs edits formatted correctly, and read well.
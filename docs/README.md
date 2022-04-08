# Documentation

This folder contains the scripts necessary to build the documentation for HugeCTR.
You can view the generated [HugeCTR documentation here](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html).

# Contributing to Docs

Follow the instructions below to be able to build the docs.

## Steps to follow:

1. To build the docs, create a developer environment for HugeCTR.  Instructions to follow as I learn them.

2. Install required documentation tools and extensions:

```shell
cd HugeCTR/docs
pip install -r requirements-doc.txt
```

4. Build the documentation:

`make clean html`

The preceding command runs Sphinx in your shell and outputs to build/html/index.html.

View docs web page by opening HTML in browser:
First navigate to /build/html/ folder, i.e., cd build/html and then run the following command:

`python -m http.server 8000`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://localhost:8000`

Now you can check if your docs edits formatted correctly, and read well.

## Decisions

### Source management

* It is a little bit of a hack, but to preserve Sphinx's expectation that all source
  files are child files and directorys of the `docs/source` directory, the `conf.py`
  configuration file copies other content, such as `sparse_operation_kit` to the
  source directory.
* One consequence of the preceding bullet is that any change to the original files,
  such as adding or removing a topic, requires a similar change to the `docs/source/index.rst`
  file.  Updating the `docs/source/index.rst` file is not automatic.

### Python API

* Until the Python API can be extracted from the C source, maintain it in the
  `docs/source/python_interface.md` file.
* When a Python class has more than one method, precede the second, third, and so on method
  with a horizontal rule by adding `***` on its own line.

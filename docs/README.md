# Documentation

This folder contains the scripts necessary to build the documentation for HugeCTR.
You can view the generated [HugeCTR documentation here](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html).

## Building the documentation

1. To build the docs, create a developer environment for HugeCTR by starting
   a container and mounting the root directory of the repository into the
   container at the `/hugectr` mount point:

   ```shell
   docker run --runtime=nvidia --rm -it --net=host --cap-add SYS_NICE \
     -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr \
     nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
   ```

1. Install required documentation tools and extensions:

   ```shell
   export HOME=/tmp
   export PATH=$HOME/.local/bin:$PATH
   python -m pip install -r docs/requirements-doc.txt
   ```

1. (Optional) Build the HPS software:

   ```shell
   pushd hierarchical_parameter_server
   python setup.py install --user
   popd
   ```

1. Build the documentation:

   ```shell
   export PYTHONPATH=$(python -c 'import site; print(site.getusersitepackages(), end="")')
   make -C docs clean html
   ```

   > **Troubleshooting Tip** To get verbose output from the Sphinx build, run
   > `make -C docs clean` and then run `sphinx-build -vv docs/source docs/build/html`.
   > When Sphinx loads libraries, the verbose output shows the path for the library.

The preceding command runs Sphinx in your shell and outputs to `docs/build/html/`.

In a shell that is outside the container, start a simple HTTP server:

`python -m http.server -d docs/build/html 8000`

Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

`https://localhost:8000`

Now you can check if your docs edits formatted correctly, and read well.

## Checking for broken links

1. Build the documentation, as described in the preceding section, but use the following command:

   ```shell
   make -C docs clean linkcheck
   ```

1. Run the link-checking script:

   ```shell
   ./docs/check_for_broken_links.sh
   ```

If there are no broken links, then the script exits with `0`.

If the script produces any output, cut and paste the `uri` value into your browser to confirm
that the link is broken.

```json
{
  "filename": "hugectr_core_features.md",
  "lineno": 88,
  "status": "broken",
  "code": 0,
  "uri": "https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/build-hadoop.sh",
  "info": "404 Client Error: Not Found for url: https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/build-hadoop.sh"
}
```

If the link is OK, and this is the case with many URLs that reference GitHub repository file headings,
then cut and paste the JSON output and add it to `docs/false_positives.json`.
Run the script again to confirm that the URL is no longer reported as a broken link.

## Decisions

### Source management: README and index files

* To preserve Sphinx's expectation that all source files are child files and directories
  of the `docs/source` directory, other content, such as the `notebooks` directory is
  copied to the source directory. You can determine which directories are copied by
  viewing `docs/source/conf.py` and looking for the `copydirs_additional_dirs` list.
  Directories are specified relative to the Sphinx source directory, `docs/source`.

* One consequence of the preceding bullet is that any change to the original files,
  such as adding or removing a topic, requires a similar change to the `docs/source/toc.yaml`
  file.  Updating the `docs/source/toc.yaml` file is not automatic.

* Because the GitHub browsing expectation is that a `README.md` file is rendered when you
  browse a directory, when a directory is copied, the `README.md` file is renamed to
  `index.md` to meet the HTML web server expectation of locating an `index.html` file
  in a directory.

### Adding notebooks

The NVIDIA TME organization wants to gather metrics for notebook usage. The proposed
method is to use a unique NVIDIA logo graphic in each notebook. For information that
shows the process of uploading the graphic and so on, see the following URL that
is internal to NVIDIA:

<https://confluence.nvidia.com/display/DL/Jupyter+Notebook+Download+Tracking>

### Adding links

TIP: When adding a link to a method or any heading that has underscores in it, repeat
the underscores in the link even though they are converted to hyphens in the HTML.

Refer to the following examples:

* `../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array`
* `./api/python_interface.md#save_params_to_files-method`

#### Docs-to-docs links

There is no concern for the GitHub browsing experience for files in the `docs/source/` directory.
You can use a relative path for the link.  For example, the following link is in the
`docs/source/hugectr_user_guide.md` file and links to the "Build HugeCTR from Source" heading
in the `docs/source/hugectr_contributor_guide.md` file:

```markdown
To build HugeCTR from scratch, refer to
[Build HugeCTR from source code](./hugectr_contributor_guide.md#build-hugectr-from-source).
```

#### Docs-to-repository links

Some files that we publish as docs, such as the `release_notes.md` file, refer readers to files
that are not published as docs. For example, we currently do not publish information from the following
directories:

* `gpu_cache`
* `onnx_converter`
* `samples`
* `tools`
* `tutorial`

To refer a reader to a README or program in one of the preceding directories, state that
the link is to the repository:

```markdown
+ **Python Script and documentation demonstrating how to analyze model files**: In this release,
we provide a script to retrieve vocabulary information from model file. Please find more details
in the README in the
[tools/model_analyzer](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.5/tools/model_analyzer)
directory of the repository.
```

The idea is to let a reader know that following the link&mdash;whether from an HTML docs page or
from browsing GitHub&mdash;results in viewing our repository on GitHub.

> TIP: In the `release_notes.md` file, use the tag such as `v3.5` instead of `master` so that
> the link is durable.

#### Links to notebooks

The notebooks are published as documentation. The few exceptions are identified in the
`docs/source/conf.py` file in the `exclude_patterns` list:

```python
exclude_patterns = [
    "notebooks/prototype_indices.ipynb",
]
```

If the document that you link from is also published as docs, such as `release_notes.md`, then
a relative path works both in the HTML docs page and in the repository browsing experience:

```markdown
+ **Support HDFS Parameter Server in Training**:
    + ...snip...
    + ...snip...
    + Added a [notebook](notebooks/training_with_hdfs.ipynb) to show how to use HugeCTR with HDFS.
```

If the document that you link from is not published as docs, such as a file in the `tools`
or `samples` directory, then either a relative path or a link to the HTML notebook is OK.

A link to the HTML notebook is like the following:

```markdown
<https://nvidia-merlin.github.io/HugeCTR/master/notebooks/continuous_training.html>
```

#### Links from notebooks to docs

Use a link to the HTML page like the following:

```markdown
<https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html>
```

> I'd like to change this in the future. My preference would be to use a relative
> path, but I need to research and change how Sphinx handles relative links.

### Python API

* Until the Python API can be extracted from the C++ source, maintain it in the
  `docs/source/api/python_interface.md` file.
* When a Python class has more than one method, precede the second, third, and so on method
  with a horizontal rule by adding `***` on its own line.

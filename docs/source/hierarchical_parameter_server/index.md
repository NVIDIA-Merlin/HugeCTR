# Hierarchical Parameter Server [Deprecated]

> **Deprecation Note** HugeCTR Hierarchical Parameter Server (HPS) has been deprecated since v25.02. Please refer to prior version if you need such features.

The Hierarchical Parameter Server (HPS) library is a native C++ library that provides
caching and hierarchical storage for embeddings.
The library is built from the GPU embedding cache and HPS database backend subcomponents.

HPS offers a flexible deployment and configuration to meet site-specific recommender system needs
and is integrated by other projects that need the ability to work with embeddings that exceed
the capacity of GPU and host memory.
Two projects that include the HPS library are the HPS plugin for TensorFlow and the
HPS backend for Triton Inference Server.

The following figure shows the relationships between the projects that use HPS,
the HPS library, and the subcomponents of the library.

<!--
<img src="/user_guide_src/hps_library.svg" alt="HPS Library and subcomponents" style="display:block;margin-left:auto;margin-right:auto;">
-->
<img src="../user_guide_src/hps_library.svg" alt="HPS Library and subcomponents" style="display:block;margin-left:auto;margin-right:auto;"/>

<div style="text-align:center"><figcaption>Fig. 1: HPS Library and subcomponents</figcaption></div>

HugeCTR Example Notebooks
=========================

.. toctree::
   :maxdepth: 1

   notebooks/hugectr_wdl_prediction.ipynb
   notebooks/hugectr2onnx_demo.ipynb
   notebooks/continuous_training.ipynb
   notebooks/ecommerce-example.ipynb
   notebooks/movie-lens-example.ipynb
   notebooks/hugectr_criteo.ipynb
   notebooks/multi_gpu_offline_inference.ipynb
   notebooks/hps_demo.ipynb

The multi-modal data example uses several notebooks to demonstrate how to use of multi-modal data (text and images)
to provide movie recommendations based on the MovieLens 25M dataset.

.. toctree::
   :maxdepth: 1

   notebooks/multi-modal-data/00-Intro
   notebooks/multi-modal-data/01-Download-Convert
   notebooks/multi-modal-data/03-Feature-Extraction-Poster
   notebooks/multi-modal-data/04-Feature-Extraction-Text
   notebooks/multi-modal-data/05-Create-Feature-Store
   notebooks/multi-modal-data/06-ETL-with-NVTabular
   notebooks/multi-modal-data/07-Training-with-HugeCTR

..
    The following notebook restarts the kernel and that causes the doc build to exit.
  - [news-example.ipynb](news-example.ipynb): Tutorial to demonstrate NVTabular for ETL the data and HugeCTR for training Deep Neural Network models on MIND dataset.

# Model Analyzer script #
The script `analyzer.py` provides statistics and information about a snapshot file of a trained (or partially trained) model.  The script computes the number of slots, and the vocabulary size (number of unique elements) of each slot. The tool requires that the model be trained using `LocalizedSlotEmbedding` to be able to extract this information from the model files.

## Usage ##
Saving a snapshot of a trained or partially trained model that uses `LocalizedSlotEmbedding` results in a directory that contains 3 files: `key`, `slot_id`, and `emb_vector`.  To run this analyzer tool on such a snapshot, simply run the following command:

```
python analyzer.py <path_to_directory>
```

Where `<path_to_directory>` is the relative or absolute path of the snapshot directory that contains the 3 files.  Below is an example of result of running this on an `ncf` model trained on the `MovieLens 20M` sample dataset.

```
$ python analyzer.py ../../samples/ncf/ncf0_sparse_400.model/
Running analysis on model files in directory: ../../samples/ncf/ncf0_sparse_400.model/
Analysis complete. Total keys: 165237
Number of slots: 2
Vocabulary size (unique keys) per slot:
Slot 0: 138493
Slot 1: 26744
```

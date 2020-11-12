## Performance Profile ##
In this directory, we compare the performance of the plugin with tensorflow original ops.

## Profile info ##
+ model: DeepFM
+ batchsize: 00000
+ batchsize_eval: 0000
+ #gpu: 1,2,4
+ slot_num: 30
+ dataset: criteo (dac.tar.gz located in ${project_name}/tools/criteo_script/)
+ vocabulary_size: 0000

## Steps ##
1. Extract dataset.
    ```shell
    $ tar zxvf dac.tar.gz
    ```
2. Preprocess dataset, fill missing value. For DeepFM, dense feature are not normalized and there are no crossed features.
    ```shell
    $ python3 preprocess.py --src_csv_path=train.txt --dst_csv_path=train.out.txt --normalize_dense=0 --feature_cross=0
    ```

3. Split dataset into train, val, test.
    ```shell
    $ head -n 36672493 train.out.txt > train
    $ tail -n 9168124 train.out.txt > valtest
    $ head -n 4584062 valtest > val
    $ tail -n 4584062 valtest > test
    ```

4. Convert dataset into TfRecord. DeepFM's dense feature is not normalized, so `--normalized=0`
    ```shell
    $ python3 txt2tfrecord.py --src_txt_name=train --dst_tfrecord_name=train.tfrecord --normalized=0 --shard_num=10 --use_multi_process=1
    $ python3 txt2tfrecord.py --src_txt_name=val --dst_tfrecord_name=val.tfrecord --normalized=0
    $ python3 txt2tfrecord.py --src_txt_name=test --dst_tfrecord_name=test.tfrecord --normalized=0
    ```

5. Do training
    + training with Plugin Embedding
    ```shell
    python3 run.py --batch_size=16384 --n_epochs=1 --distribute_keys=1 --gpus 0 1 3 4 --embedding_type='localized' --vocabulary_size=1737710 --embedding_vec_size=10 --slot_num=26 --batch_size_eval-=4
    ```
    
    + training with Original Embedding
    ```shell
    python3 run.py --batch_size=16384 --n_epochs=1 --distribute_keys=0 --gpus 0 1 3 4 --embedding_type='localized' --vocabulary_size=1737710 --embedding_vec_size=10 --slot_num=26 --batch_size_eval=4 --which_embedding=OriginalEmbedding
    ```

## Results ##


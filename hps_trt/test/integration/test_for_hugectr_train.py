import hugectr
from hugectr.tools import DataGeneratorParams, DataGenerator


def test_data_generation():
    data_generator_params = DataGeneratorParams(
        format=hugectr.DataReaderType_t.Parquet,
        label_dim=1,
        dense_dim=13,
        num_slot=26,
        i64_input_key=True,
        nnz_array=[1 for _ in range(26)],
        source="./data_parquet/file_list.txt",
        eval_source="./data_parquet/file_list_test.txt",
        slot_size_array=[10000 for _ in range(26)],
        check_type=hugectr.Check_t.Non,
        dist_type=hugectr.Distribution_t.PowerLaw,
        power_law_type=hugectr.PowerLaw_t.Short,
        num_files=16,
        eval_num_files=4,
        num_samples_per_file=40960,
    )
    data_generator = DataGenerator(data_generator_params)
    data_generator.generate()


def test_train():
    solver = hugectr.CreateSolver(
        model_name="dlrm",
        max_eval_batches=160,
        batchsize_eval=1024,
        batchsize=1024,
        lr=0.001,
        vvgpu=[[0]],
        repeat_dataset=True,
        use_mixed_precision=True,
        use_cuda_graph=True,
        scaler=1024,
        i64_input_key=True,
    )
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Parquet,
        source=["./data_parquet/file_list.txt"],
        eval_source="./data_parquet/file_list_test.txt",
        slot_size_array=[10000 for _ in range(26)],
        check_type=hugectr.Check_t.Non,
    )
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.Adam,
        update_type=hugectr.Update_t.Global,
        beta1=0.9,
        beta2=0.999,
        epsilon=0.0001,
    )

    model = hugectr.Model(solver, reader, optimizer)
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="numerical_features",
            data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("keys", 1, True, 26)],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=5000,
            embedding_vec_size=128,
            combiner="mean",
            sparse_embedding_name="sparse_embedding1",
            bottom_name="keys",
            optimizer=optimizer,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.MLP,
            bottom_names=["numerical_features"],
            top_names=["mlp1"],
            num_outputs=[512, 256, 128],
            act_type=hugectr.Activation_t.Relu,
            use_bias=True,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Interaction,
            bottom_names=["mlp1", "sparse_embedding1"],
            top_names=["interaction1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.MLP,
            bottom_names=["interaction1"],
            top_names=["mlp2"],
            num_outputs=[1024, 1024, 512, 256, 1],
            use_bias=True,
            activations=[
                hugectr.Activation_t.Relu,
                hugectr.Activation_t.Relu,
                hugectr.Activation_t.Relu,
                hugectr.Activation_t.Relu,
                hugectr.Activation_t.Non,
            ],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["mlp2", "label"],
            top_names=["loss"],
        )
    )
    model.graph_to_json("dlrm_hugectr_graph.json")
    model.compile()
    model.summary()
    model.fit(
        max_iter=1200,
        display=200,
        eval_interval=1000,
        snapshot=1000,
        snapshot_prefix="dlrm_hugectr",
    )

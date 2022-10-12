import numpy as np
import tensorflow as tf

from sparse_operation_kit import experiment as sok


dim = 128
vocab_size = 1024 * 128
batch = 8192

optimizers = [
    tf.train.GradientDescentOptimizer(learning_rate=1.0),
    tf.train.MomentumOptimizer(learning_rate=1.0, momentum=0.9),
    tf.train.AdagradOptimizer(learning_rate=1.0, initial_accumulator_value=0.1),
    tf.train.AdadeltaOptimizer(learning_rate=1.0),
    tf.keras.optimizers.SGD(lr=1.0),
    tf.keras.optimizers.SGD(lr=1.0, momentum=0.9),
    tf.keras.optimizers.Adamax(learning_rate=1.0, beta_1=0.9, beta_2=0.999),
    tf.keras.optimizers.Adadelta(learning_rate=1.0),
    tf.keras.optimizers.Adagrad(learning_rate=1.0),
    tf.keras.optimizers.Ftrl(learning_rate=1.0),
    # tf.keras.optimizers.RMSprop(learning_rate=1.0),
    # tf.keras.optimizers.Adam(learning_rate=1.0, beta_1=0.9, beta_2=0.999),
    # tf.keras.optimizers.Nadam(learning_rate=1.0),
    # tf.train.AdamOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.999),
    # tf.train.FtrlOptimizer(learning_rate=1.0),
]

for optimizer in optimizers:
    indices = tf.placeholder(shape=[None], dtype=tf.int64)
    weight = tf.placeholder(shape=[None, dim], dtype=tf.float32)

    sok_var = sok.DynamicVariable(dimension=dim)
    emb = tf.nn.embedding_lookup(sok_var, indices)
    emb_mul = emb * weight
    loss = tf.reduce_sum(emb_mul)
    grads = tf.gradients(loss, [sok_var])

    sok_optimizer = sok.OptimizerWrapper(optimizer)
    train_op = sok_optimizer.apply_gradients(zip(grads, [sok_var]))

    init_val = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    tf_var = tf.Variable(
        initial_value=[[0.0] * dim for _ in range(vocab_size)],
        shape=[vocab_size, dim],
        use_resource=True,
    )
    assign_op = tf_var.assign(init_val)
    tf_emb = tf.nn.embedding_lookup(tf_var, indices)
    tf_emb_mul = tf_emb * weight
    tf_loss = tf.reduce_sum(tf_emb_mul)
    tf_grads = tf.gradients(tf_loss, [tf_var])
    tf_optimizer = optimizer
    tf_train_op = tf_optimizer.apply_gradients(zip(tf_grads, [tf_var]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        indices_val = [idx for idx in range(vocab_size)]
        table_val = sess.run([emb], feed_dict={indices: indices_val})[0]
        sess.run([assign_op], feed_dict={init_val: table_val})

        for i in range(100):
            print("---------------------Iter %d---------------------" % i)
            num = np.random.randint(1, batch + 1, 1)[0]
            indices_val = np.random.randint(0, vocab_size, num).astype(np.int64)
            weight_val = np.random.rand(num, dim).astype(np.float32)
            loss_val, _ = sess.run(
                [loss, train_op], feed_dict={indices: indices_val, weight: weight_val}
            )
            tf_loss_val, _ = sess.run(
                [tf_loss, tf_train_op], feed_dict={indices: indices_val, weight: weight_val}
            )
            print("loss:", loss_val, "tf_loss:", tf_loss_val)

        indices_val = [idx for idx in range(vocab_size)]
        sok_result = sess.run([emb], feed_dict={indices: indices_val})[0]
        tf_result = sess.run([tf_var])[0]
        print("sok_result[0, :]")
        print(sok_result[0, :])
        print("tf_result[0, :]")
        print(tf_result[0, :])
        err = ((sok_result - tf_result) ** 2.0).mean()
        assert err < 1e-6

    print("[SOK INFO] Test variable successfully with optimizer:", end=" ")
    print(optimizer)

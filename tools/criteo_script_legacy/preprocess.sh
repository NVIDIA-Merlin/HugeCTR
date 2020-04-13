#!/bin/bash

#decompresion
tar zxvf dac.tar.gz

head -n 36672493 train.txt > train
tail -n 9168124 train.txt > valtest
head -n 4584062 valtest > val
tail -n 4584062 valtest > test

# will produce train.out val.out test.out
perl preprocess.pl train val test


# may need to shuffle train.out & val.out

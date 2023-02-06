#!/usr/bin/perl
#
# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

use warnings;
use strict;
use Getopt::Std;
use autodie qw(open close);


my $MIN_CNT   = 6;
my $NCOL      = 40;
my $EMPYT_FEA = '-';


@ARGV == 3
  or die "usage: $0 train val test\n";

open my $fh_train, '<', $ARGV[0];
my $fea2id = generate_mapping($fh_train);
close $fh_train;

for my $file (@ARGV) {
    open my $fh_in,  '<', $file;
    open my $fh_out, '>', "${file}.out";
    do_mapping($fea2id, $fh_in, $fh_out);
    close $fh_in;
    close $fh_out;
}




sub generate_mapping {
    my ($fh) = @_;

    my %feature;
    for my $i (1 .. $NCOL - 1) {
        $feature{$i}{$EMPYT_FEA} = $MIN_CNT;
    }

    while (<$fh>) {
        chomp;
        my @fields = split(/\t/, $_, -1);
        @fields == $NCOL
          or die "#fields = " . scalar(@fields);

        for my $i (1 .. $#fields) {
            my $fea = $fields[$i];
            $fea = $EMPYT_FEA if $fea eq '';
            ++$feature{$i}{$fea};
        }
    }
    if ($MIN_CNT > 1) {
        for my $i (1 .. $NCOL - 1) {
            while (my ($k, $v) = each %{$feature{$i}}) {
                delete $feature{$i}{$k} if $v < $MIN_CNT;
            }
        }
    }

    my %fea2id;
    my $id = 1;
    for my $i (1 .. $NCOL - 1) {
        my @fea =
          sort { $feature{$i}{$b} <=> $feature{$i}{$a} or $a cmp $b }
          keys %{$feature{$i}};
        for my $fea (@fea) {
            $fea2id{$i}{$fea} = $id;
            ++$id;
        }
    }

    return \%fea2id;
}


sub do_mapping {
    my ($fea2id, $fh_in, $fh_out) = @_;
    while (<$fh_in>) {
        chomp;
        my @fields = split(/\t/, $_, -1);
        @fields == $NCOL
          or die "#fields = " . scalar(@fields);

        print {$fh_out} "$fields[0]";
        for my $i (1 .. $#fields) {
            my $fea = $fields[$i];
            if ($fea eq '' || !(exists $fea2id->{$i}{$fea})) {
                $fea = $EMPYT_FEA;
            }
            print {$fh_out} " ", $fea2id->{$i}{$fea};
        }
        print {$fh_out} "\n";
    }
}


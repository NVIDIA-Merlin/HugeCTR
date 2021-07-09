#! /bin/bash

libs=`find /usr/local/cuda/lib/ -name "*.so*" | rev | cut -d'/' -f 1 | rev | cut -d'.' -f 1`
libs+=`find /usr/local/cuda/lib64/ -name "*.so*" | rev | cut -d'/' -f 1 | rev | cut -d'.' -f 1`
for lib in $libs
do
  lib_condas=`find /opt/conda/ -name "$lib*.*"`
  for lib_conda in $lib_condas
  do
    if [[ "$lib_conda" == *".so"* || "$lib_conda" == *".a"* ]]; then
      if [[ "$lib_conda" != *"libnpp"* ]]; then
        rm -rfv $lib_conda
      fi
    fi
  done
done

for package in "$@"
do
  libs=`find /usr/lib/x86_64-linux-gnu/ -name "*$package*" | rev | cut -d'/' -f 1 | rev | cut -d'.' -f 1`
  for lib in $libs
  do
    lib_condas=`find /opt/conda/ -name "$lib*.so*"`
    for lib_conda in $lib_condas
    do
      if [[ "$lib_conda" == *".so"* || "$lib_conda" == *".a"* ]]; then
        rm -rfv $lib_conda
      fi
    done
  done

  headers=`find /usr/include/ -name "*$package*" | rev | cut -d'/' -f 1 | rev | cut -d'.' -f 1`
  for header in $headers
  do
    header_condas=`find /opt/conda/ -name "$header*.h"`
    for header_conda in $header_condas
    do
      rm -rfv $header_conda
    done
  done
done

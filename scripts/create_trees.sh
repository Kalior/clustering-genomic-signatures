#!/bin/sh
# Place this script in the same folder that contains as your all.sh
# script and the profiles folder.

# Directory where the created trees will be moved
PST_DIR=/home/neon/Documents/exjobb/clustering-genomic-signatures/trees


echo "Removing existing trees"
rm $PST_DIR/*

echo "Generating new trees"
./all.sh 1 3

echo "Copying trees"
cp profiles/*.tree $PST_DIR

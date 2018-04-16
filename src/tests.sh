#!/bin/sh
parameters=(12 24 48 96) #128 192 252 516 768)

output=../results
image_folder=../images/frobenius

make

mv $output $output.old
date | tee $output

for p in "${parameters[@]}"
do
  echo "Frobenius with $p parameters:" | tee -a $output
  mkdir -p ${image_folder}_$p
  python3.6 -u test_distance_function.py --frobenius --directory ../trees_$p --out-directory ${image_folder}_$p |& tee -a $output
done

#!/bin/sh

mkdir -p output

python3 -u classifier.py --dataset=mnist --model=linear                    > output/linear
python3 -u classifier.py --dataset=mnist --model=factorized_linear         > output/factorized_linear
python3 -u classifier.py --dataset=mnist --model=neural_network --size=256 > output/neural_network
python3 -u classifier.py --dataset=mnist --model=kitchen_sink --size=256   > output/kitchen_sink

for i in 16 32 64 128 256 512 1024; do
    python3 -u classifier.py --dataset=mnist --model=neural_network --size=$i --epoch=25 > output/neural_network.$i
done

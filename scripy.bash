#!/bin/zsh

tasks=( 'Join_SemiDistributive' 'Meet_SemiDistributive' )
modes=( 'strong' 'weak' )
seeds=( 102 106 270 )

for task in SemiDistributive
do
    for seed in 102 106 270
    do
        for mode in strong weak
        do
            echo "$mode"
            python main.py extract-motifs --task="$task" --generalisation_mode="$mode" --n_epochs=10 --seed="$seed"
        done
    done
done
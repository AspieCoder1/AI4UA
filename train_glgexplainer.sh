#!/bin/zsh

tasks=( 'Distributive' 'Modular' 'QuasiCancellitive' )
modes=( 'strong' 'weak' )
seeds=( '102' '106' '270' )
num_prototypes=(2 4 6 8)
num_motifs=(1 2 3 4 5)

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
      for mode in "${modes[@]}"; do
        for n_motifs in "${num_motifs[@]}"; do
          for n_prototypes in "${num_prototypes[@]}"; do
            python main.py train-explainer --task="$task" --generalisation_mode="$mode" --seed="$seed" --n-motifs="$n_motifs" --n-prototypes="$n_prototypes"
          done
        done
      done
  done
done
#!/bin/zsh

tasks=( 'QuasiCancellitive' )
modes=( 'strong' 'weak' )
seeds=( '102' '106' '270' )

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
      for mode in "${modes[@]}"; do
        python main.py extract-motifs --task="$task" --generalisation_mode="$mode" --n_epochs=100 --seed="$seed"
      done
  done
done
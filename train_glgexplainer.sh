#!/bin/zsh

tasks=( 'Distributive' 'Modular' )
modes=( 'strong' 'weak' )
seeds=( '102' '106' '270' )

for task in "${tasks[@]}"; do
  for seed in "${seeds[@]}"; do
      for mode in "${modes[@]}"; do
        python main.py train-explainer --task="$task" --generalisation_mode="$mode" --seed="$seed"
      done
  done
done
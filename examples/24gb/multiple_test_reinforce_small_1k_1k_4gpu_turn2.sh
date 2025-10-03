usage() { echo "Usage: $0 <jl|lua|ml|r|rkt>"; exit 1; }

case "$1" in
  jl)  lang=humaneval-jl; bsize=159 ;;
  lua) lang=humaneval-lua; bsize=161 ;;
  ml)  lang=humaneval-ml; bsize=155 ;;
  r)   lang=humaneval-r; bsize=161 ;;
  rkt) lang=humaneval-rkt; bsize=161 ;;
  *) usage ;;
esac

export OMP_NUM_THREADS=4

torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=test@jusjinuk/MultiPL-E-fixed:${lang} \
    train_data.prompts_per_rollout=${bsize} \
    train_data.responses_per_prompt=4 \
    test_data.prompts_per_rollout=${bsize} \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.max_length_per_device=2560 \
    rollout.max_turns=2 \
    rollout.train_sampling_params.max_new_tokens=1024 \
    rollout.max_new_tokens_from_turn2=1024 \
    rollout.env_path=envs/multiple_turn2.py \
    adv.estimator=reinforce \
    trainer.project=MultiPL-E-$1 \
    trainer.experiment_name=multiple-$1-qwen3-4b-thinking-2507-reinforce-1k-1k-4gpu-turn2-pdb \
    trainer.save_freq=20 \
    trainer.n_epochs=80 \
    rollout.save_trajectories=true \
    rollout.trajectories_save_freq=1 \

    # Reduce training memory peak
    actor.use_liger_kernel=true \ 
    # Higher: Faster Rollout (SGLang can use more memory for caching, etc) but be careful with OOM
    rollout.gpu_memory_utilization=0.7


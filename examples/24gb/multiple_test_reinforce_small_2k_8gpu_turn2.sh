usage() { echo "Usage: $0 <jl|lua|ml|r|rkt>"; exit 1; }

case "$1" in
  jl)  lang=humaneval-jl; bsize=159 ;;
  lua) lang=humaneval-lua; bsize=161 ;;
  ml)  lang=humaneval-ml; bsize=155 ;;
  r)   lang=humaneval-r; bsize=161 ;;
  rkt) lang=humaneval-rkt; bsize=161 ;;
  *) usage ;;
esac

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=test@jusjinuk/MultiPL-E-fixed:${lang} \
    train_data.prompts_per_rollout=${bsize} \
    train_data.responses_per_prompt=4 \
    test_data.prompts_per_rollout=${bsize} \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.sp_size=4 \
    actor.max_length_per_device=2048 \
    rollout.max_turns=2 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    rollout.max_new_tokens_from_turn2=1024 \
    rollout.env_path=envs/multiple_turn2.py \
    adv.estimator=reinforce \
    trainer.project=MultiPL-E-$1 \
    trainer.experiment_name=multiple-$1-qwen3-4b-thinking-2507-reinforce-4k-1k-8gpu-turn2 \
    trainer.save_freq=20 \
    trainer.n_epochs=100 \
    rollout.save_trajectories=true \
    rollout.trajectories_save_freq=1
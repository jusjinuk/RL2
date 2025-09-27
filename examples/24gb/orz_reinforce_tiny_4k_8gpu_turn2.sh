torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=jusjinuk/OpenReasonerZero-with-system \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=8 \
    test_data.path=jusjinuk/OlympiadBench-with-system \
    actor.model_name=Qwen/Qwen3-1.7B \
    actor.sp_size=4 \
    actor.max_length_per_device=4096 \
    rollout.max_turns=2 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    rollout.env_path=envs/orz_turn2.py \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen3-1.7b-reinforce-4k-8gpu-turn2 \
    trainer.test_freq=8 \
    trainer.save_freq=32 \
    rollout.save_trajectories=true \
    rollout.trajectories_save_freq=1
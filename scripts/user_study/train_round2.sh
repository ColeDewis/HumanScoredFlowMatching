# Examples:
# bash scripts/user_study/train_round2.sh USER_NAME PROTOCOL

DEBUG=False
save_ckpt=True

user_id=${1}
protocol=${2} # "single" or "multiple"
alg_name=rtc_flow_match_rgb_sirius
task_name=franka_banana_rgb_sirius
config_name=${alg_name}
addition_info=0322
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data_sirius/user_study/${user_id}/${protocol}/output/round2/${exp_name}_seed${seed}"

dataset_path=data_sirius/user_study/${user_id}/${protocol}/combined_round2.zarr

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd flow_policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=${dataset_path}



                                

export ACCELERATE_USE_FSDP=1 
export FSDP_CPU_RAM_EFFICIENT_LOADING=1 
export NCCL_IB_DISABLE=1 

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch fsdp_qlora_train_0521.py \
    --config fsdp_config.yaml
conda create -n fsdp python=3.10 -y
conda activate fsdp
pip install "torch==2.2.2" tensorboard
pip install  --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3" "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2" "trl==0.8.6" "peft==0.10.0" loguru wandb
huggingface-cli login --token <hf-token>
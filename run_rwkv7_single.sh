NVCC_PREPEND_FLAGS='-ccbin /home/linuxbrew/.linuxbrew/bin/gcc-11':
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --standalone --nproc_per_node=1 train_rwkv7_single.py "$@"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=

# export OMP_NUM_THREADS=1
# export WANDB_USERNAME=
# export WANDB_API_KEY=
# export WANDB_PROJECT=
# export WANDB_ENTITY=


python3 timm_train.py \
    /dune/DATASETS/ILSVRC2012 \
    -b 128 \
    --model fixed_simplex_resnet50 \
    --lr 0.6 \
    --warmup-epochs 5 \
    --epochs 240 \
    --weight-decay 1e-4 \
    --sched cosine \
    --reprob 0.4 \
    --recount 3 \
    --remode pixel \
    --aa rand-m7-mstd0.5-inc1 \
    -j 12 \
    --amp \
    --dist-bn reduce \
    --output ./output




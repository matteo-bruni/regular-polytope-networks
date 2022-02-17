# Regular Polytope Networks

PyTorch code for the paper "Regular Polytope Networks" (TNNL)


## CIFAR

we leverage the implementation from https://github.com/kuangliu/pytorch-cifar


```
python cifar.py --prototype simplex --dataset cifar10 --net resnet50
```

## ImageNet

we leverage the timm library available here: https://github.com/rwightman/pytorch-image-models

```
sh example_timm_imagenet.sh
```

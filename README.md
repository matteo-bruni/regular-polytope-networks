# Regular Polytope Networks

This repo contains the code of "Regular Polytope Networks" (TNNL)

Our code relies on the following impementations:
 - https://github.com/kuangliu/pytorch-cifar for Cifar
 - https://github.com/rwightman/pytorch-image-models timm for ImageNet experiments
 

Refer to our paper for more details: 
 - https://ieeexplore.ieee.org/document/9358981
 - https://arxiv.org/abs/2103.15632


## Abstract

> Neural networks are widely used as a model for classification in a large variety of tasks. 
Typically, a learnable transformation (i.e., the classifier) is placed at the end of such models 
returning a value for each class used for classification. 
This transformation plays an important role in determining how the generated features change 
during the learning process. In this work, we argue that this transformation not only can be 
fixed (i.e., set as nontrainable) with no loss of accuracy and with a reduction in memory usage, 
but it can also be used to learn stationary and maximally separated embeddings. We show that the 
stationarity of the embedding and its maximal separated representation can be theoretically justified 
by setting the weights of the fixed classifier to values taken from the coordinate vertices of the 
three regular polytopes available in Rd, namely, the d-Simplex, the d-Cube, and the d-Orthoplex. 
These regular polytopes have the maximal amount of symmetry that can be exploited to generate stationary 
features angularly centered around their corresponding fixed weights. Our approach improves and broadens 
the concept of a fixed classifier, recently proposed by Hoffer et al., to a larger class of fixed 
classifier models. Experimental results confirm the theoretical analysis, the generalization capability, 
the faster convergence, and the improved performance of the proposed method.


## Preparation

Install required dependencies
```
pip install -r requirements.txt
```


## Usage


### Train RepoNet on CIFAR

```
python cifar.py --prototype simplex --dataset cifar10 --net resnet50
```

### Train RepoNet on ImageNet

the following script use timm to train on ImageNet, specify the required paths

```
sh example_timm_imagenet.sh /path/to/imagenet/folder /output/path
```

## Authors

- Federico Pernici <federico.pernici at unifi.it> [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/FedPernici.svg?style=social&label=FedPernici)](https://twitter.com/FedPernici)
- Matteo Bruni <matteo.bruni at unifi.it>
- Claudio Baecchi <claudio.baecchi at unifi.it>
- Alberto Del Bimbo <alberto.delbimbo at unifi.it>


## Citing

Please kindly cite our paper if this repository is helpful.
```
@ARTICLE{9358981,
  author={Pernici, Federico and Bruni, Matteo and Baecchi, Claudio and Bimbo, Alberto Del},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Regular Polytope Networks}, 
  year={2021},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2021.3056762}
}
```
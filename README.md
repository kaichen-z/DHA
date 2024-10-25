# DHA: End-to-End Joint Optimization of Data Augmentation Policy, Hyper-parameter and Architecture

[Paper Link](https://arxiv.org/abs/2109.05765)

We propose ***DHA***, which achieves joint optimization of Data augmentation
policy, Hyper-parameter and Architecture.

🌟 ***End-toE-End Traning***: First to efficiently and jointly realize Data Augmentation, Hyper-parameter Optimization, and Neural Architecture Search in an
end-to-end manner without retraining.

🌟 ***State-of-the-art***: State-of-the-art accuracy on ImageNet with both cell-based and Mobilenet-like architecture search space

🌟 ***Findings***: Demonstrate the advantages of doing joint-training over optimizing each AutoML component in sequence

## 💾 juDataset Preparation
[[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)]
[[Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)]
[[IMAGENET](https://image-net.org/download.php)]
[[SPORTS8](http://vision.stanford.edu/lijiali/event_dataset/)]
[[MIT67](http://web.mit.edu/torralba/www/indoor.html)]
[[FLOWERS102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)]


## 💻 Usage

### Training with flowers102 dataset

Training normal two-stage ISTA Model
```bash
python search_darts.py --conf_path conf/ista/flowers102_ista.json --Running ista_nor
```

Training normal single-stage ISTA Model
```bash
python search_darts.py --conf_path conf/ista/flowers102_ista_single.json --Running ista_single_nor
```

Training DHA Model
```bash
python search_darts.py --conf_path conf/ista/flowers102_ista_single.json --Running ista_single_doal
```

## 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
@article{zhou2021dha,
  title={Dha: End-to-end joint optimization of data augmentation policy, hyper-parameter and architecture},
  author={Zhou, Kaichen and Hong, Lanqing and Hu, Shoukang and Zhou, Fengwei and Ru, Binxin and Feng, Jiashi and Li, Zhenguo},
  journal={arXiv preprint arXiv:2109.05765},
  year={2021}
}
```
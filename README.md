# Geometry Distributions

### [Project Page](https://1zb.github.io/GeomDist/) | [Paper (arXiv)](https://arxiv.org/abs/2411.16076)

### :bullettrain_front: Training

```
torchrun --nproc_per_node=4 main.py --blr 5e-7 --output_dir output/loong --log_dir output/loong --data_path shapes/loong.obj
```

### :balloon: Inference

```
python infer.py --pth output/loong/checkpoint-999.pth --target Gaussian --num-steps 64 --output samples/loong --N 10000000
```

### :floppy_disk: Datasets
https://huggingface.co/datasets/Zbalpha/shapes

### :briefcase: Checkpoints
https://huggingface.co/Zbalpha/geom_dist_ckpt

## :e-mail: Contact

Contact [Biao Zhang](mailto:biao.zhang@kaust.edu.sa) ([@1zb](https://github.com/1zb)) if you have any further questions. This repository is for academic research use only.

## :blue_book: Citation

```bibtex
@article{zhang2024geometry,
  title={Geometry Distributions},
  author={Zhang, Biao and Ren, Jing and Wonka, Peter},
  journal={arXiv preprint arXiv:2411.16076},
  year={2024}
}
```

python infer.py --pth ../checkpoints/FAUST_features/checkpoint-5.pth --target Gaussian --num-steps 64 --output ../samples/FAUST_features_scaling/ --N 1000000 --feature-dim 1


torchrun --nproc_per_node=1 main.py --blr 5e-7 --batch_size 131072 --accum_iter 2 --output_dir ../checkpoints/FAUST_features --log_dir ../logs/FAUST_features --data_path ../MPI-FAUST/training/registrations/tr_reg_000.ply --feature_path ../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt --feature_interpolation nearest-neighbor --resume ../checkpoints/FAUST_features/checkpoint-25.pth

torchrun --nproc_per_node=1 main.py --blr 5e-7 --batch_size 131072 --accum_iter 2 --output_dir ../checkpoints/spot_color --log_dir ../logs/spot_color --data_path ../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/spot/spot_uv_normalized.obj --texture_path ../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/spot/spot_by_keenan.png --resume ../checkpoints/spot_color/checkpoint-45.pth


python infer.py --pth ../checkpoints/spot_color/checkpoint-45.pth --target Gaussian --num-steps 64 --output ../samples/shapes/spot_color --N 1000000 --texture

torchrun --nproc_per_node=1 main.py --blr 5e-7 --batch_size 131072 --accum_iter 2 --output_dir ../checkpoints/FAUST_vertexid --log_dir ../logs/FAUST_vertexid --data_path ../MPI-FAUST/training/registrations/tr_reg_000.ply --feature_path ../SMPL_python_v.1.1.0/smpl_template_indices.txt --feature_interpolation barycentric

python infer.py --pth ../checkpoints/FAUST_scaling/checkpoint-105.pth --target Gaussian --num-steps 64 --output ../samples/FAUST_scaling/ --N 1000000 --feature-dim 1 --max_categorical_feature 6889

torchrun --nproc_per_node=1 main.py --blr 5e-7 --batch_size 131072 --accum_iter 2 --output_dir ../checkpoints/FAUST_scaling --log_dir ../logs/FAUST_scaling --data_path ../MPI-FAUST/training/registrations/tr_reg_000.ply --feature_path ../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt --feature_interpolation nearest-neighbor --resume ../checkpoints/FAUST_scaling/checkpoint-5.pth

torchrun --nproc_per_node=1 main.py --blr 5e-7 --batch_size 87382 --accum_iter 3 --output_dir ../checkpoints/FAUST_scaling_depth8 --log_dir ../logs/FAUST_scaling_depth8 --data_path ../MPI-FAUST/training/registrations/tr_reg_000.ply --feature_path ../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt --feature_interpolation nearest-neighbor --depth 8

python infer.py --pth ../checkpoints/FAUST_scaling_depth8/checkpoint-20.pth --target Gaussian --num-steps 64 --output ../samples/FAUST_scaling_depth8/ --N 1000000 --feature-dim 1 --max_categorical_feature 23 --depth 8
# icon-gen

This repository contains code for training a 2D image generation GAN. It also contains helper scripts for image augmentation and label sorting.

## Models

The generator and discriminator can be found under `icon_gen`. Hyperparameters can be set in a config file such as `config/spell.yaml`

To start training:

```
python icon_gen/train.py --config config/spell.yaml
```

### Scripts

The `scripts/` directory contains helper scripts that can be used to prepare data for training.

#### cut_borders

Cuts the borders around images in `dir` by `border_width` pixels on each side and saves the edited copies to `out_dir`.

#### flip

Flips images vertically and horizontally according to flags and saves new flipped images to input `dir`.

#### sorter

Launches a simple UI to help sorting images into classes. It will take as input a `dir` that should contain all the image data and already have class subdirectories created. Each direct subdirectory inside of `dir` is treated as a subclass and will appear as a class that can be chosen for an image.

Each image will then pop up in order prompting which class it is. Upon selection of a class the image will be moved to the subdirectory with the chosen class name. 




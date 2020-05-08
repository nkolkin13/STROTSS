# Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS)
Code for the paper https://arxiv.org/abs/1904.12785 (CVPR 2019)

UPDATE 5/8/2020: David Futschik (https://github.com/futscdav) very kindly pointed out a bug in the feature extraction pipeline where the images were not properly normalized with imagenet's mean and standard deviation for each color channel. Fixing this dramatically improves results in many cases. He also has implemented a much faster and more memory efficient version of strotts (https://github.com/futscdav/strotss), it doesn't allow for spatial guidance yet but I'm planning on incorporating his improvements into this repo soon so that the faster version is available for spatial guidance as well.

## Dependencies:
* python3 >= 3.5
* pytorch >= 1.0
* imageio >= 2.2
* numpy >= 1.1

## Usage:
### Unconstrained Style Transfer:

```
python3 styleTransfer.py {PATH_TO_CONTENT} {PATH_TO_STYLE} {CONTENT_WEIGHT}
```

The default content weight is 1.0 (for the images provided my personal favorite is 0.5, but generally 1.0 works well for most inputs). The content weight is actually multiplied by 16, see section 2.5 of paper for explanation. 

The resolution of the output can be set on line 80 of styleTransfer.py; the current scale is 5, and produces outputs that are 512 pixels on the long side, setting it to 4 or 6 will produce outputs that are 256 or 1024 pixels on the long side respectively, most GPUs will run out of memory for settings of this variable above 6.

The output will appear in the same folder as 'styleTransfer.py' and be named 'output.png'

### Spatially Guidaed Style Transfer:

```
python3 styleTransfer.py {PATH_TO_CONTENT} {PATH_TO_STYLE} {CONTENT_WEIGHT} -gr {PATH_TO_CONTENT_GUIDANCE} {PATH_TO_STYLE_GUIDANCE}
```

guidance should take the form of two masks such as these:


Content Mask           |  Style Mask
:-------------------------:|:-------------------------:
<img height="200" src='https://github.com/nkolkin13/STROTSS/blob/master/content_guidance.jpg?raw=true'> |  <img height="200" src='https://github.com/nkolkin13/STROTSS/blob/master/style_guidance.jpg?raw=true'>


where regions that you wish to map onto each other have the same color.

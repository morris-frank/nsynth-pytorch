# Nsynth-PyTorch

![](https://github.com/morris-frank/nsynth-pytorch/workflows/pytest/badge.svg)

This is reimplementation of the NSynth model as described in _Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders_ [[arxiv:1704.01279](http://arxiv.org/abs/1704.01279)].

The original TensorFlow v1 code can be found under in [github:tensorflow/magenta](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth).

## Requirements
Python package requirements for this code are [`torch >= 1.3.1`](https://pypi.org/project/torch/1.3.1/) and [`librosa>=0.7.1`](https://pypi.org/project/librosa/0.7.1).
Further to load audio with librosa you need [`libsndfile`](https://github.com/erikd/libsndfile) which most systems should already have.

To replicate the original experiment you will have to download the NSynth dataset under https://magenta.tensorflow.org/datasets/nsynth#files in `json/wav` format.

## Train
A training script can be found in `train.py` which will train the model with the same parameters as in the original paper.

```bash
python ./train.py --datadir SOME_PATH/nsynth/ --device=cuda:0 --nbatch=32
```
The required argument _datadir_ should be de full path to the NSynth dataset. (The path should contain folders `nsynth-test`, `nsynth-train` and `nsynth-valid` each of whom have to contain the folder `/audio/` and the file `examples.json`.)

The other default arguments will give the same setting as in the original paper.
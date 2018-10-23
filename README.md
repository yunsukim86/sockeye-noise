## Denoising Autoencoder in Sockeye

This version of Sockeye contains codes to train a denoising autoencoder for sequences. It includes the following artificial noises for source side:

- Insertion of frequent tokens
- Deletion of tokens
- Permutation of tokens with a limited distance

If you use this code, please cite:

- Yunsu Kim, Jiahui Geng and Hermann Ney. [Improving Unsupervised Word-by-Word Translation Using Language Model and Denoising Autoencoder](https://www-i6.informatik.rwth-aachen.de/publications/download/1075/Kim-EMNLP-2018.pdf). EMNLP 2018.
- Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post. [Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). arXiv preprint.

### Installation

```bash
> pip install -r requirements/requirements.txt
> pip install .
```
after cloning the repository from git.

If you want to run on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by
running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).

### Usage

To train a denoising autoencoder, turn on `--source-noise-train` with detailed noise options (`--source-noise-insertion`, `--source-noise-insertion-vocab`, `--source-noise-deletion`, `--source-noise-permutation`). Please put the **same training data** for both source and target sides and also the **same validation data** for both sides. Optionally, you can also switch on `--source-noise-validation` to evaluate your models on a noisy validation set during the training. Example:
```bash
> python -m sockeye.train -s {training_data} \
                          -t {training_data} \
                          -vs {validation_data} \
                          -vt {validation_data} \
                          --source-noise-train \
                          --source-noise-permutation 3 \
                          --source-noise-deletion 0.1 \
                          --source-noise-insertion 0.1 \
                          --source-noise-insertion-vocab 50 \
                          .... (other options)
```
Denoising with a trained model can be done with `sockeye.translate` module in the same way as translating an input sentence. You can use all other modules provided by Sockeye on denoising autoencoder, e.g. sharding the training data (`sockeye.prepare_data`) or model averaging (`sockeye.average`). Please refer to the [Sockeye documentation](https://awslabs.github.io/sockeye/) for details.


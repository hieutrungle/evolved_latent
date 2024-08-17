==============
evolved_latent
==============


.. image:: https://img.shields.io/pypi/v/evolved_latent.svg
        :target: https://pypi.python.org/pypi/evolved_latent

.. image:: https://img.shields.io/travis/hieutrungle/evolved_latent.svg
        :target: https://travis-ci.com/hieutrungle/evolved_latent

.. image:: https://readthedocs.org/projects/evolved-latent/badge/?version=latest
        :target: https://evolved-latent.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT
* Documentation: https://evolved-latent.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `briggySmalls/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`briggySmalls/cookiecutter-pypackage`: https://github.com/briggySmalls/cookiecutter-pypackage

## Installation

poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/cu124
poetry add --source pytorch-gpu-src torch torchvision torchaudio

## Commands

Train Autoencoder

```bash
poetry run main --command train_autoencoder --num_epochs 250 --batch_size 40 \
        --grad_accum_steps 1 --workers 8 --autoencoder_type resnet_norm
```

Train EvoLatent

```bash
poetry run main --command train_evo --num_epochs 250 --batch_size 40 \
        --grad_accum_steps 1 --workers 8 --autoencoder_type resnet_norm \
        --autoencoder_checkpoint ./logs/ResNetNormAutoencoder_20240812-205311/ \
        --evo_type transformer --evo_hidden_size 256 --evo_num_layers 3
```

Combine Autoencoder and EvoLatent

```bash
poetry run main --command train_evo --num_epochs 250 --batch_size 40 \
        --grad_accum_steps 1 --workers 8 --autoencoder_type resnet_norm \
        --autoencoder_checkpoint ./logs/ResNetNormAutoencoder_20240812-205311/ \
        --evo_type transformer --evo_hidden_size 64 --evo_num_layers 6 \
        --evo_checkpoint ./logs/EvolvedLatentTransformer_20240815-163019/
```

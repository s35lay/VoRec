# VoRec
This repository contains a implementation of our "**VoRec: Enhancing Recommendation with Voronoi Diagram in Hyperbolic Space**".

## Environment Setup
1. torch 2.0.0
2. Python 3.10.14
3. geoopt (`$ pip install git+https://github.com/geoopt/geoopt.git`)
4. numpy
5. scipy
6. pandas
7. tqdm

## Guideline

### data

We provide one dataset, ciao.

```adj_csr.npz``` adj matrix built for training gcn.

```item_tag_matrix.npz``` items attributes matrix. 

```train.pkl``` train set.

```test.pkl``` test set.

```user_item_list.pkl``` user-item dict for the complete dataset.

```implication.pt``` the hierarchies between tags.

### models

The implementation of model(```model.py```).
code to implement Hyperbolic gcn (```encoders.py, hyp_layers.py```).

### utils

```data_generator.py``` read and organize data.

```helper.py``` some method for helping preprocess data or set seeds and devices.

```sampler.py``` a parallel sampler to sample batches for training.

```train_utils.py``` read and parse the config arguments.

## Example to run the codes

```
python run.py
python run.py > logs/ciao.log
```

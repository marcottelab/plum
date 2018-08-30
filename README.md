# plum ![](images/plum_small.png)

***Generalizable phylogenetic latent variable models for ancestral network reconstruction***

![](https://zenodo.org/badge/DOI/10.5281/zenodo.1406146.svg)
---------------

## Citations

This code package is in support of Liebeskind *et al.* 2018

For more reading see:
 - [Koch *et al.* 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5515301/)
 - [Roy *et al.* 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668358/)
 - [Bykova *et al.* 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3676395/)
 - [Zhang and Moret 2011](https://link.springer.com/chapter/10.1007/978-3-642-21260-4_33)

## About

Phylogenetic latent variable models (PLVM, pronounced "plum" because in the Latin alphabet, uppercase 'u' was written 'V') are 
a class of stochastic evolutionary models that are similar to the standard models used in phylogenetics, except instead of 
assuming the data at the tips of the tree are known, they model uncertainty in the data. The evolving character is therefore
"latent," i.e. not directly observed.

## Installation

### Requirements

```python
abc
scipy
numpy
pandas
cython
dendropy
scikit-learn
```

### Procedure
```bash
git clone
cd plum
```

#### If you want to use it in place
```bash
python setup.py build_ext --inplace
```

Note that if you use this option you will have to add plum and its subdirectories to your PATH and PYTHONPATH

#### If you want to install it
```bash
python setup.py build_ext
python setup.py build
python setup.py install
```

#### If you want to install locally
```bash
python setup.py build_ext
python setup.py build
python setup.py install --user
```

## Usage

#### Pipeline
If you have some data from several species that reports on the presence/absence of some discrete variable, the typical usage of this package 
will be something like this:
 - Format your data
 - Choose a model
 - Train and test the model
 - If you like the look of your tests, predict on the entire set

If you want to interact with `plum` in a more object oriented fashion, see the ipython notebook tutorial in notebooks/

#### Data format

You can look at data formats for data, tree, and parameter files in testdata/
A few extra notes:
 - In the data files, the column called `data` can have any name (call it something that describes your input feature(s)) but leave the
 other file names the same.
 - The column `state` contains the labels. This column is used differently by training and prediction scripts. When training, it is
 ignored while calculating the scores on the tree and used only for recall-precision evaluation. But when predicting, this column will
 be used to clamp the value of that node to whatever state you specify. If you don't want this latter behavior, remove values from this
 column during prediction.
 - For training purposes, when you're using smaller data files, the input does not need to be sorted, but when you predict on the entire
 dataset, it's best to sort the data by your ID columns. This is because, if you don't, the program will have to sort using pandas, 
 which is much slower than bash. See example below in **Prediction** section
 - param files support C++ style comments, using "//" before the comment.
 
#### Split your data into training and test sets

First, navigate to testdata, which has some data sets and parameter files for you to try out

```bash
python ../bin/split_plum_training-data.py --infile training-data_small_multivariate.csv --fold 2
```

This will write:
```bash
    test0_training-data_small_multivariate.csv
    test1_training-data_small_multivariate.csv
    train0_training-data_small_multivariate.csv
    train1_training-data_small_multivariate.csv
```

#### Choose an error model

**Choices**

 - Univariate models
    - Gaussian
    - GaussianMix
    - Gumbel
    - GumbelMix
    - Gamma
    - GammaMix
    - Cauchy
    - CauchyGumbel
 - Multivariate models
    - MultivariateGaussian
    - MultivariateGaussianMixture
    - MultivariateGaussianDiagCov
    
We're fitting multivariate data, so let's choose MultivariateGaussian. We can use the param file in testdata for
starting parameters and bounds

#### Fit the model and test it on hold-out data

```bash
python ../bin/fit_plum_simulated-annealing.py --training_data train0_training-data_small_multivariate.csv \
--test_data test0_training-data_small_multivariate.csv --treefile unikont_tree.nhx \
--paramfile mvgaussian.param --job_name test0 --criterion likelihood --start_temp 1.0 \
--alpha .3 --temp_steps 5 --mutation_sd .3
```

This will fit a PLVM using the MultivariateGaussian and the standard TwoState Markov model (both of which are specified
in the param file). We've asked it to use likelihood as the fitting criterion. Note that we're also using simulated 
annealing parameters that will fit the model very quickly. If you want a better fit (you do), increase alpha to .9 and 
temp_steps to 10 or 20.

After fitting, there will be four new files in testdata/
```bash
    test0_params.txt
    test0_resultsDF.csv
    test0_testPRC.csv
    test0_trainPRC.csv
```
 
_params.txt contains the fit parameters and comments that tell you about the fit. Here's what mine looks like:

```bash
// Training set train0_training-data_small_multivariate.csv
// Criterion: likelihood
// Training best score: -263.890842938
// Test best average precision score: 0.0881418942533
# Error Model
Name: MultivariateGaussian
Params: mean0=[0.19033966293739998, 0.01117662571935278];sigma0=[[1.0, -0.5478893507714943], [-0.5478893507714943, 0.56304722043422]];mean1=[0.05504330013754061, -0.06514285186851157];sigma1=[[0.6530801758523388, 0.5071404781048408], [0.5071404781048408, 1.0]]

# Markov Model
Name: TwoState
Params: alpha=0.01;beta=0.654341332952
```

Obviously this is a terrible fit, but we don't expect much on this tiny dataset with such an insufficient fitting procedure.

_resultsDF.csv contains the results of the fit to the training data with FDR calculation
_testPRC.csv and _trainPRC.csv contain the information necessary to make a precision-recall curve on the results for the test and training data

**Note:** we just fit on a single subset, but you'll probably want to fit on multiple subsets, with higher-fold cross-validation and/or multiple replicates

## Predict using your model

Now we can predict using our terrible model above, but first we should sort our data:

```bash
head -n1 training-data_small_multivariate.csv >> training-data_small_multivariate.sorted.csv
grep -v "data" training-data_small_multivariate.csv | sort -t, -k1,2 >> training-data_small_multivariate.sorted.csv

python ../bin/predict_plum.py --datafile training-data_small_multivariate.sorted.csv --treefile unikont_tree.nhx \
--paramfile test0_params.txt --outfile test0_prediction.csv --as_sorted
```

This produces the finished results with predictions at interior nodes

| ID1         | ID2         | node             | P_1              | P_event          | known_state | 
|-------------|-------------|------------------|------------------|------------------|-------------| 
| ENOG4102NDK | ENOG4104K0S | Unikonts         | 0.0119673220697  | 0.0              |             | 
| ENOG4102NDK | ENOG4104K0S | Euarchontoglires | 0.00221244218068 | 0.00631175279897 |             | 
| ENOG4102NDK | ENOG4104K0S | Dm               | 0.0120762595083  | 0.00189947541066 |             | 
| ENOG4102NDK | ENOG4104K0S | Xl               | 0.0194883296436  | 0.0109641346639  |             | 
| ENOG4102NDK | ENOG4104K0S | Mm               | 0.00295623738354 | 0.00074379520286 |             | 
| ENOG4102NDK | ENOG4104K0S | Hs               | 0.0              | 0.00221244218068 | 0           | 
| ENOG4102NDK | ENOG4104K0S | Dd               | 0.0144985921622  | 0.00253127009253 |             | 
| ENOG4102NDK | ENOG4104K0S | Sp               | 0.0124215955469  | 0.00260440235976 |             | 

`P_1` is the score at each node while `P_event` is the absolute difference between the score at this node and at
the parental node. 

**Note:** We used training-data_small_multivariate.sorted.csv as our data to predict on, which had values in the `state`
column, so as dicussed above, the scores are clamped to these values as you can see for the `Hs` node above.

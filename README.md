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
python ../bin/split_plum_training-data.py --infile training-data_small_multivariate.csv --fold 2 --random_seed 2112
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
--alpha .3 --temp_steps 5 --mutation_sd .3 --random_seed 2001
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
 
_params.txt contains the fit parameters and comments that tell you about the fit. Here's what it looks like:

```bash// Training set train0_training-data_small_multivariate.csv
// Criterion: likelihood
// Training best score: -242.37015280420815
// Test best average precision score: 0.12853174603174602
# Error Model
Name: MultivariateGaussian
Params: mean0=[0.09271652962259244, 0.15133451340886617];sigma0=[[1.0, -0.8780918018636823], [-0.8780918018636823, 1.0]];mean1=[0.39235386064996314, 0.19154441337520783];sigma1=[[0.4682611089433504, 0.3554577482681259], [0.3554577482681259, 0.6223605063177067]]

# Markov Model
Name: TwoState
Params: alpha=0.24970275115394253;beta=0.19244346977974788
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

This produces the finished results with predictions at interior nodes. The top of the file looks like this

|ID1        |ID2        |node            |P_1                |P_event             |known_state|
|-----------|-----------|------------- --|-------------------|--------------------|-----------|
|ENOG4102NDK|ENOG4104K0S|Unikonts        |0.4277910907412827 |0.0                 |           |
|ENOG4102NDK|ENOG4104K0S|Opisthokonts    |0.4030955029650736 |0.02469558777620906 |           |
|ENOG4102NDK|ENOG4104K0S|Sc              |0.4958592145259373 |0.09276371156086366 |           |
|ENOG4102NDK|ENOG4104K0S|Eumetazoa       |0.38170506539670246|0.021390437568371168|           |
|ENOG4102NDK|ENOG4104K0S|Bilateria       |0.37947037251042925|0.002234692886273204|           |
|ENOG4102NDK|ENOG4104K0S|Deuterostomes   |0.3700163676208032 |0.009454004889626055|           |
|ENOG4102NDK|ENOG4104K0S|Tetrapods       |0.3392019911771062 |0.030814376443697   |           |
|ENOG4102NDK|ENOG4104K0S|Xl              |0.693888308641325  |0.3546863174642188  |           |
|ENOG4102NDK|ENOG4104K0S|Euarchontoglires|0.09574458021994871|0.2434574109571575  |           |
|ENOG4102NDK|ENOG4104K0S|Mm              |0.11400615359796604|0.01826157337801733 |           |
|ENOG4102NDK|ENOG4104K0S|Hs              |0.0                |0.09574458021994871 |0          |


`P_1` is the score at each node while `P_event` is the absolute difference between the score at this node and at
the parental node. 

**Note:** We used training-data_small_multivariate.sorted.csv as our data to predict on, which had values in the `state`
column, so as dicussed above, the scores are clamped to these values as you can see for the `Hs` node above.

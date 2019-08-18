import gc
import os
import argparse
import numpy as np
import pandas as pd
import plum.util.data
import plum.models.ErrorModels
import plum.models.MarkovModels
import plum.training.likelihood
import plum.training.recall_precision
#import plum.training.classifier_likelihood

parser = argparse.ArgumentParser("Fit a PLVM model on training data and then test it on new data")
parser.add_argument("--training_data", required=True, help="Tidy csv with columns ID1, ID2, species, data, state")
parser.add_argument("--test_data", required=True, help="Tidy csv with columns ID1, ID2, species, data, state")
parser.add_argument("--treefile", required=True, help="A newick treefile")
parser.add_argument("--paramfile", required=True, help="Parameter file determining model and starting params")
parser.add_argument("--job_name", required=True, help="Name used for formatting outfiles")
parser.add_argument("--criterion", default="recall-precision", choices=["likelihood","recall-precision","classifier-likelihood"], help="Which fitting criterion to use")
parser.add_argument("--weight_errormodel_by_prior",action='store_true',help='''For recall-precision criterion. Whether to weight likelihood
                                                                                         under the error model by the prior from the stationary frequencies''')

parser.add_argument("--start_temp", default=1., type=float, help="Starting temperature of simulated annealing algorithm")
parser.add_argument("--alpha", default=.9, type=float, help="Decrementing factor for temperature schedule")
parser.add_argument("--temp_steps", default=10, type=int, help="Number of steps to take at each temperature")
parser.add_argument("--mutation_sd", default=.5, type=float, help="Standard deviation of the sampling distribution (a Gaussian)")

parser.add_argument("--random_seed", default=False, type=int, help="Random seed for simulated annealing algorithm")

args = parser.parse_args()

assert os.path.exists(args.training_data), "File not found: {}".format(args.training_data)
assert os.path.exists(args.test_data), "File not found: {}".format(args.test_data)
assert os.path.exists(args.treefile), "File not found: {}".format(args.treefile)
assert os.path.exists(args.paramfile), "File not found: {}".format(args.paramfile)

if args.weight_errormodel_by_prior == True:
    assert args.criterion == 'recall-precision', "`weight_errormodel_by_prior` flag can only be used with criterion = recall-precision"

## Read in and create models
print("Reading parameter file")
em,mm = plum.util.data.get_models_from_file(args.paramfile)


## Fit models by recall-precision or likelihood
if args.criterion == 'recall-precision':
    model = plum.training.recall_precision.simulated_annealing(error_model=em,
                                                                markov_model=mm,
                                                                tree=args.treefile,
                                                                data=args.training_data,
                                                                prior_weighted=args.weight_errormodel_by_prior,
                                                                start_temp=args.start_temp,
                                                                alpha=args.alpha,
                                                                temp_steps=args.temp_steps,
                                                                mutation_sd=args.mutation_sd,
                                                                random_seed=args.random_seed)
    print("Fitting model on {}".format(args.training_data))
    model.fit()
    fit_params = model.best_params
    best_score = model.best_aps
                                                                
elif args.criterion == "likelihood":
    likelihood_model = plum.training.likelihood.simulated_annealing(error_model=em,
                                                                markov_model=mm,
                                                                tree=args.treefile,
                                                                data=args.training_data,
                                                                start_temp=args.start_temp,
                                                                alpha=args.alpha,
                                                                temp_steps=args.temp_steps,
                                                                mutation_sd=args.mutation_sd,
                                                                random_seed=args.random_seed)
    print("Fitting model on {}".format(args.training_data))
    likelihood_model.fit()
    fit_params = likelihood_model.best_params
    best_score = likelihood_model.best_logL
    
    model = plum.training.recall_precision.plumRecallPrecision(error_model = eval(str(em))(**likelihood_model.errorModelParams),
                                                               markov_model = eval(str(mm))(**likelihood_model.markovModelParams),
                                                               tree=args.treefile, data=args.training_data)
                                                               
else:
    likelihood_model = plum.training.classifier_likelihood.simulated_annealing(error_model=em,
                                                                markov_model=mm,
                                                                tree=args.treefile,
                                                                data=args.training_data,
                                                                start_temp=args.start_temp,
                                                                alpha=args.alpha,
                                                                temp_steps=args.temp_steps,
                                                                mutation_sd=args.mutation_sd,
                                                                random_seed=args.random_seed)
    print("Fitting model on {}".format(args.training_data))
    likelihood_model.fit()
    fit_params = likelihood_model.best_params
    best_score = likelihood_model.best_classL
    
    model = plum.training.recall_precision.plumRecallPrecision(error_model = eval(str(em))(**likelihood_model.errorModelParams),
                                                               markov_model = eval(str(mm))(**likelihood_model.markovModelParams),
                                                               tree=args.treefile, data=args.training_data)

    
## Create and write training PR curves
train_prc = model.precision_recall_DF
train_prc["dataset"] = "{}_train".format(args.job_name)
train_prc.to_csv(args.job_name + "_trainPRC" + ".csv", index=False)
train_prc = None

## Test model on hold-out set
print("Testing fitted model {} on test data".format(args.job_name))
fit_em = eval(str(em))(**model.errorModelParams)
fit_mm = eval(str(mm))(**model.markovModelParams)

model = None
gc.collect()

test_model = plum.training.recall_precision.plumRecallPrecision(error_model=fit_em, markov_model=fit_mm,data=args.test_data,tree=args.treefile)

## Write parameter values and scores
with open(args.job_name + "_params.txt",'w') as out:
    out.write("// Training set {}\n".format(args.training_data))
    out.write("// Criterion: {}\n".format(args.criterion))
    out.write("// Training best score: {}\n".format(best_score))
    out.write("// Test best average precision score: {}\n".format(test_model.aps))
    out.write("# Error Model\n")
    out.write("Name: {}\n".format(str(em).split(".")[-1]))
    if test_model.is_multivariate:
        out.write( "Params: {}\n\n".format( ';'.join( ["=".join([i,str(fit_params[i].tolist())]) for i in em.freeParams] ) ) )
    else:
        out.write( "Params: {}\n\n".format( ';'.join( ["=".join([i,str(fit_params[i])]) for i in em.freeParams] ) ) )
    out.write("# Markov Model\n")
    out.write("Name: {}\n".format(str(mm).split(".")[-1]))
    out.write( "Params: {}\n\n".format( ';'.join( ["=".join([i,str(fit_params[i])]) for i in mm.freeParams] ) ) )

        
## Write test precision-recall curve
test_prc = test_model.precision_recall_DF
test_prc["dataset"] = "{}_test".format(args.job_name)
test_prc.to_csv(args.job_name + "_testPRC" + ".csv", index=False)

resultsDF = test_model.results_DF
resultsDF.to_csv( args.job_name + "_resultsDF.csv",index=False)

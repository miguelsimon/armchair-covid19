# Overview

This contains a layman's attempts at understanding covid-19 using quantitative measures.

<img src="media/daily_deaths.png" alt="daily_deaths" width="400"/>

[The jupyter notebook](sir.ipynb) contains the actual work.

* [Goals](#goals)
  * [Disease model](#Disease-model)
  * [Infer model parameters](#Infer-model-parameters)
* [Usage](#usage)

## Goals

### Disease model

Create a disease dynamics model to simulate disease progression, based on the [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).

Full explanation in [sir.ipynb](sir.ipynb), code for this is mainly in [sir.py](sir.py).

We can use the model by plugging in parameter estimates we get from publications on covid-19 eg. [R0 estimates](https://www.ncbi.nlm.nih.gov/pubmed/32097725) and using it to make disease forecasts to guide our intuition.

A few months ago when I was growing confused at the news about covid-19: I decided to get some clarity by learning some basic epidemiology:
* I wrote a trivial numerical integrator for the SIR model
* plugged in R0 and mortality estimates (and their error bars) appearing in peer-reviewed publications
* plugged in the numbers for Madrid
* compared a scenario with lockdown to a no-lockdown scenario

and came up with this, most scenarios for the plausible R0s and fatality rates were pretty scary:

<img src="media/comparison.png" alt="comparison" width="400"/>

### Infer model parameters

Instead of guessing at the model parameters we want to estimate them by building a statistical model and fitting it to observed daily fatalities: this is what qualified epidemiologists should be doing to advise governments on good policy.

Full explanation in [inference.ipynb](inference.ipynb), code for this builds on the previous section and is mainly in [inference.py](inference.py).

If the model is close enough to reality (very doubtful in this case) it should be useful to forecast disease progression to some extent, and to retrospectively understand how the disease was spreading months ago.

To do this:
* I've built a [bayesian hierarchical model](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling) using the disease dynamics of the SIR model
* I've build what is probably the slowest, gnarliest [Metropolis-Hastings Markov chain monte carlo sampler](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) ever constructed to perform inference, might get fancier and attempt Hamiltonian monte carlo using [pymc3](https://docs.pymc.io/) in future
* TODO: apply it to a real data set

## Usage

You'll need make and a reasonably modern version of python 3.

* `make env_ok` sets up the virtual env
* `make test` runs typecheckers, linters and unit tests
* `make fmt` runs code formatters

To run the jupyter notebooks in the correct context, simply `make run_notebook`.

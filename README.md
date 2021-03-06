# DeepOneClass

This project implements the code for a Deep Learning model performing One-Class 
Classification for an anomaly detection objective. This implementation use the libraries 
**tensorflow** and **tensorlayer**.

## Training and evaluation

Run the training script with desired flags:
```shell
python source_files/training.py \
--type ocsvm \
--epochs 1000 \
--learning_rate 0.005 \
--batch_size 128 \
--c 0.1 \
--model_dir ../tmp/estimator \
--class_nbr 1
```

Run validation script:
```shell
python validation.py \
--type ocsvm \
--batch_size 128 \
--c 0.1 \
--model_dir ../tmp/estimator \
--class_nbr 1 \
```

See `source_files/flags.py` for more flags options.

## Copyright and Licensing

The project is released under the GNU Affero GPL v3.0 License, which gives you 
the following rights in summary:

|**Permissions**       |**Limitations**    |**Conditions**                                      |
|--------------------- |------------------ |--------------------------------------------------- |
|:+1: *Commercial use* |:x: *Liability*    |:information_source: *License and copyright notice* |
|:+1: *Modification*   |:x: *Warranty*     |:information_source: *State changes*                |
|:+1: *Distribution*   |                   |:information_source: *Disclose source*              |
|:+1: *Patent use*     |                   |:information_source: *Network use is distribution*  |
|:+1: *Private use*    |                   |:information_source: *Same license*                 |


## Contributing guidelines

Please have a look to the [Contributing Guidelines](CONTRIBUTING.md) first.

We follow the "fork-and-pull" Git workflow.

1. **Fork** the repo on GitHub
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## Project Installation

```shell
########################################################
# =============== Create a virtualenv  =============== #
########################################################

## Install virtualenv if necessary
pip install virtualenv

## Then create a virtualenv called venv inside
virtualenv venv

########################################################
# ============= Activate the virtualenv  ============= #
########################################################

# Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate.bat

########################################################
# ============ Install the dependencies  ============= #
########################################################

# Linux:
pip install -r requirements.txt

# Windows:
pip install lib_bin\scikit_image-0.13.1-cp36-cp36m-win_amd64.whl # This librabry must be pre-compiled
pip install -r requirements.txt
```



## Launching the Unit Tests


```shell
########################################################
# ============= Activate the virtualenv  ============= #
########################################################

# Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate.bat

########################################################
# =============== Launching the tests  =============== #
########################################################

python unittests.py
```

This should give you a similar output:
```
###########################
DEBUG:tensorflow:OneClassCNN:output => Shape: (64, 16, 16, 256) - Mean: 5.008926e-01 - Std: 0.002996 - Min: 0.494600 - Max: 0.507075
DEBUG:tensorflow:OneClassCNN:logits => Shape: (64, 16, 16, 256) - Mean: 3.570643e-03 - Std: 0.011985 - Min: -0.021600 - Max: 0.028301

----------------------------------------------------------------------
Ran 3 tests in 5.934s

OK
```


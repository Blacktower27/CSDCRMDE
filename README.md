This is the code accompanying the paper: "Container Scheduling with Dynamic Computing Resource for Microservice Deployment in Edge Computing"

## Description
This simplified code implements a SAC-based algorithm to solve the container-based online microservice scheduling problem with Dynamic Computing Resource.

## Dependencies

You just need to install **torch**, numpy, random, csv, argparse by pip or conda

## Usage

Firstly, you could set the config of the environment including node number, task number, etc. in 

```
config.py 
```

Then, you could run and train the model by

```
python sac_gru.py
```
You can run other baseline code in the same way.

Finally, the training results could be find under `--log`

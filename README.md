# Polymerase Misincorporation Kinetic Simulations


## Installation in a Conda environment
An Anaconda environment can be used to easily install the required Python dependendcies.
The *conda_sim.sh* bash script will install the required Python dependencies.

```bash
bash conda_sim.sh sim
source activate sim
```

## Usage

```bash
python sim.py [input.csv] [# of MC Error Iterations] [Polymerase Model]
```

Example data set can be found in 'Example' and replicated with

```bash
python sim.py example_input.csv 200 E
```
This reads in the rate constants and errors from the example_input.csv file and performed 200 MC error iterations using the rate constants for human polymerase epsilon. 
E = pol epsilon
B = pol Beta
T7 = T7

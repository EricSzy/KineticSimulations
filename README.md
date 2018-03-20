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
python sim.py [input.csv] [# of MC Error Iterations] [Polymerase Model] [Incorporation Model]
```

A template of the input csv can be created with:

```bash
python sim.py -setup
```

Example data set can be found in 'Example'. The example_input.csv was used with the following command in this example. 

```bash
python sim.py example_input.csv 200 E Dual
```
This reads in the rate constants and errors from the example_input.csv file and performed 200 MC error iterations using the rate constants for human polymerase epsilon. 


Polymerases:

E = pol epsilon

B = pol Beta

T7 = T7

Incorporaiton Model:

Dual = Both ES1 and ES2 can be incorporated.

ES1 = Only ES1 can be incorporated.

ES2 = Only ES2 can be incorporated.

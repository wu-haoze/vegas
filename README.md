# vegas
Recently, Graph Neural Networks (GNNs) have been applied for scheduling jobs
over clusters, achieving better performance than hand-crafted heuristics.
Despite their impressive performance, concerns remain over whether these
GNN-based job schedulers meet users' expectations about other important
properties, such as strategy-proofness, sharing incentive, and stability.
We present *vegas*, the first formal verification engine for GNN-based job
schedulers. To achieve scalability, *vegas* combines abstractions,
refinements, solvers, and proof transfer.

For more information about the tool, please check out our paper: https://arxiv.org/pdf/2203.03153.pdf


# Download
The latest version of vegas is available on https://github.com/NeuralNetworkVerification/Marabou.

# Setup and installation

In order to install and use vegas, first install the required python packages.
We recommend running the tool in a virtual environment. You could create a
virtual environment using *virtualenv*. Note that the GNN-scheduler, Decima,
which we verify as a proof-of-concept, needs to be run with python version <= 3.7.
```
virtualenv -p python3.7 py3
source py3/bin/activate 
```

One some system, you need to reinstall pip for the particular virtual environment:
```
curl -sS https://bootstrap.pypa.io/get-pip.py  -o get-pip.py 
python get-pip.py --force-reinstall
```

After creating the virtual environment, run
```
pip install -r requirements.txt
```

Next, the verification backend specific to GNN-schedulers are
implemented on top of the Marabou framework for neural network verification:
```
git clone https://github.com/anwu1219/Marabou
cd Marabou
git fetch origin vegas
git checkout vegas
mkdir build
cd build
cmake ../ -DENABLE_GUROBI=ON
make -j12
```

If the installation is successful, a binary called Marabou should be generated
in the folder. You could either add the binary to the system path or copy it to
the root directory of the repo:

```
cp Marabou ../../verifier
cd ../../
```

# Define verification queries

One can define a verification query through the Python front-end of vegas.
Currently we support properties with respect to a particular job profile.
Thus, generating a verification query involves the following three steps:
1. Load the GNN verifier;
2. Load a initial job profile;
3. Define the specification.
4. Dump the verification query.

These steps can be done through the Python API. In particular, our front-end
contains methods to define linear constraints on the node features of any
nodes in the job profile, as well as on the decision variables of the scheduler.
An example is provided in src/strategy-proofness-example.py

# Reproduce the results

The 298 verification queries used in the paper,
which are generated using the method described above, are contained
in oopsla22/ipqs/ They can be directly checked from the command line
using the *verifier* binary that we created. 

The binary takes in command options that correspond to the configurations described in the paper.
- F: forward analysis only
```
  --milp
```
- F+B1: performing 1 iteration of forward and backward analysis
```
  --milp --backward
```
- F+BC: performing forward and backward analysis to convergence
```
  --milp --backward --converge
```
- A+F+BC: node abstraction scheme on top of F+BC
```
  --milp --backward --converge  --relax
```

Additionally, to enable parallelization (processing N neurons in the same layer in parallel):
```
  --num-workers=N
```

To set a time limit of T seconds:
```
   --timeout 3600
```

To dump a summary file containing the runtime and the verification result:
```
--summary-file PATH/TO/FILE
```


For example, to solve with 16 threads a particular benchmark decima_jobs5_execs5_rep11_step20.ipq,
which checks the single-step strategy-proofness property on a job profile with
5 jobs with the forward-backward analysis to convergence:
```
./verifier --milp --backward --converge --num-workers=16 --input-query=oopsla22/ipqs/decima_jobs5_execs5_rep11_step20.ipq --summary-file=summary.txt  --timeout=600

```

The solver should be able to solve the query within 3 minutes and you should
 see *unsat* in the last line of the output, which means the property is verified.
Moreover, you should see a file called "summary.txt" generated at the current directory
with its content being
"UNSAT [RUNTIME] [other metrics] [other metrics]" 

Note that it will take much longer to solve the same query if we use the
forward abstract interpretation only mode:
```
./verifier --milp --num-workers=16 --input-query oopsla22/ipqs/decima_jobs5_execs5_rep11_step20.ipq

```

To reproduce the results in the paper, one can run the different configurations
on all the verification queries in the folder by running in the oopsla22 folder the script:
```
./runAllBenchmarks.sh
```

This will create four folders corresponding to the configurations and each folder
will contain the summary files of running the configuration on the benchmarks.


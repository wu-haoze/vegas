import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import sys
import time
sys.path.insert(0, '/home/haozewu/Projects/vmware/robust-decima/resources/decima-sim-master')
sys.path.insert(0, '/home/haozewu/Projects/vmware/Marabou')

from verification_util import load_benchmark_env,run_with_trace_recording, close_competition

from src_verify.DecimaNetwork import DecimaNetwork

from spark_env import env as orig_env
from deterministic_actor_agent import ActorAgent
from param import *
from msg_passing_path import get_unfinished_nodes_summ_mat

import attacks
import importlib
import time

DURATION_MIN = 0.5*0.001943# 200/100000.
DURATION_MAX = 1.2*16.2513 # 300000./100000.
TASK_MIN = 1/200.
TASK_MAX = 1.2*2.965 # 500./200.

def main():
    # get benchmark
    num_jobs = args.num_init_dags
    num_executors = args.exec_cap
    rep = args.rep
    trial = args.trial
    steps = args.steps
    backwardProp = args.backward
    abstraction = args.abstraction
    dumpQueryFile = args.dump_query_file

    print("{} jobs, {} executors, rep {}, trial {}".format(num_jobs, num_executors,
          rep, trial))

    decimaNetwork = DecimaNetwork("gcn.tf", args, steps=steps,
                                  checkNodeScore=True, checkJobExecScore=False,
                                  verbosityLevel=2)
    candidateSteps = decimaNetwork.candidateSteps

    for step in candidateSteps:
        if step > decimaNetwork.allSteps / 3:
            exit(10)
        filename = (f"verification_results/decima_jobs{num_jobs}_execs{num_executors}" +
                    f"_rep{rep}_step{step}_backProp{backwardProp}_trial{trial}.txt")
        print(f"Writing result to {filename}")
        if os.path.isfile(filename):
            continue

        decimaNetwork.setInitialState(step)
        secondJobIdx = decimaNetwork.getSecondJobIndex()

        numFrontier = 0
        for nodeIdx in decimaNetwork.jobToNodes[secondJobIdx]:
            if nodeIdx in decimaNetwork.validNodes:
                numFrontier += 1

        if abstraction:
            decimaNetwork.createDecimaNetwork(secondJobIdx)
        else:
            decimaNetwork.createDecimaNetwork()

        # Only increase the node features in the second job
        for nodeIdx in decimaNetwork.jobToNodes[secondJobIdx]:
            if nodeIdx in decimaNetwork.validNodes:
                duration = decimaNetwork.nodeInputs[nodeIdx][3]
                task = decimaNetwork.nodeInputs[nodeIdx][4]
                decimaNetwork.setNodeFeatureBounds(nodeIdx,
                                                   upper=min(DURATION_MAX, duration * 2),
                                                   featureIdx=3)
                decimaNetwork.setNodeFeatureBounds(nodeIdx, upper=min(TASK_MAX, task * 2),
                                                   featureIdx=4)
                decimaNetwork.setPerTaskDurationLowerBound(nodeIdx, duration / task)

        decimaNetwork.addConstraintJobMax(secondJobIdx)

        if dumpQueryFile == None:
            start = time.time()
            result, vals = decimaNetwork.solve(dumpBounds=False,
                                               backwardBoundTightening=backwardProp,
                                               deleteQueryFile=True,
                                               timeout=300)
            finish = time.time()
            runTime = finish-start

            with open(filename, 'w') as out_file:
                out_file.write(f"{result} {runTime}\n")
        else:
            result, vals = decimaNetwork.solve(dumpBounds=False,
                                               backwardBoundTightening=backwardProp,
                                               deleteQueryFile=False,
                                               timeout=300,
                                               queryFileName=
                                               os.path.join
                                               (dumpQueryFile,
                                                f"decima_jobs{num_jobs}_" +
                                                f"execs{num_executors}" +
                                                f"_rep{rep}_step{step}_10x.ipq"))
    exit(10)

if __name__ == "__main__":
    main()

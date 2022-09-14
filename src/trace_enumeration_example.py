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

from src_verify.MultiStepVerification import MultiStepVerification
from src_verify.ScheduleTemplate import ScheduleTemplate

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
    proofTransfer = args.proof_transfer
    complete = args.complete

    print("{} jobs, {} executors, rep {}, trial {}".format(num_jobs, num_executors,
          rep, trial))

    msv = MultiStepVerification(None, args)
    trace = msv.getTrace()

    filename = (f"tree/decima_jobs{num_jobs}_execs{num_executors}" +
                f"_rep{rep}_step{steps}_trial{trial}_proofTransfer{proofTransfer}_naive{complete}")

    print(f"Writing result to {filename}")
    if os.path.isfile(filename):
        exit(0)

    jobs = set()
    nodeToSchedule = None
    jobToPerturb = None
    for s, (node, job) in enumerate(trace[steps:]):
        if s >= 5 and (job not in jobs):
            nodeToSchedule = node
            jobToPerturb = job
            break
        jobs.add(job)

    print(f"job to perturb: {jobToPerturb}")

    if jobToPerturb == None:
        return

    thisMsv = MultiStepVerification(None, args, agent=msv.agent)

    for nodeIdx in thisMsv.decimaNetwork.jobToNodes[jobToPerturb]:
        if nodeIdx in thisMsv.decimaNetwork.validNodes:
            duration = thisMsv.decimaNetwork.nodeInputs[nodeIdx][3]
            task = thisMsv.decimaNetwork.nodeInputs[nodeIdx][4]
            thisMsv.addPerturbation(nodeIdx, upper=min(duration * 10, DURATION_MAX), featureIdx=3)
            thisMsv.addPerturbation(nodeIdx, upper=min(task * 10, TASK_MAX), featureIdx=4)

    initialSchedule = [n for (n,j) in trace[:steps]]
    traces, times = thisMsv.getTree(initialSchedule, 5)

    schedule = [n for (n,j) in trace]

    with open(filename, 'w') as out_file:
        out_file.write(f"job to nodes: {msv.initialJobToNodes}\n")
        out_file.write(f"job perturbed: {jobToPerturb}\n")
        out_file.write(f"ground truth trace: {schedule}\n")
        out_file.write(f"num traces: {len(traces)}\n")
        for i, t in enumerate(traces):
            out_file.write(t + f" {times[i]}\n")

if __name__ == "__main__":
    main()

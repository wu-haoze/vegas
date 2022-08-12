import pickle as pkl
from os import path

import numpy as np
import tensorflow as tf

from deterministic_actor_agent import ActorAgent
from spark_env import env as orig_env

BENCHMARK_DIR = 'benchmark/'
MULTI_STEP_BENCHMARK_DIR = 'multi-step-benchmarks/states/'

def load_benchmark_env(n_jobs, n_execs, rep):
    """
    Load a stored environment for benchmarking
    """
    envpath = path.join(BENCHMARK_DIR, f'{n_jobs}-{n_execs}/{rep}.pkl')
    with open(envpath, 'rb') as fin:
        env = pkl.load(fin)
    return env

def load_multi_step_benchmark_env(n_jobs, n_execs, rep):
    """
    Load a stored environment for benchmarking
    """
    envpath = path.join(MULTI_STEP_BENCHMARK_DIR, f'{n_jobs}-{n_execs}/{rep}.pkl')
    with open(envpath, 'rb') as fin:
        env = pkl.load(fin)
    return env

#def parsePropertyFile(filename):
#    with open(filename, 'r') as in_file:


def close_node_competition(trace_node_probs, delta=0.9):
    """
    Find steps along the trajectory in which the difference
    between the most likely and second-most likely jobs isn't
    too large.
    """
    interesting_steps = []
    for i, job_probs in enumerate(trace_node_probs):
        if len(job_probs)<=1:
            continue
        job_probs = sorted(job_probs, key=lambda x: -x)
        if job_probs[0]-job_probs[1]<=delta:
            interesting_steps.append(i)
    return interesting_steps

# Copied over from https://github.com/mahmoods01/robust-decima/blob/31f281ce51840278cde52efbc86dcdc4cfd3c224/resources/decima-sim-master/sequential-strategy-proofness-invalidation.py
def close_competition(trace_job_probs, delta=0.90):
    """
    Find steps along the trajectory in which the difference
    between the most likely and second-most likely jobs isn't
    too large.
    """
    interesting_steps = []
    for i, job_probs in enumerate(trace_job_probs):
        if len(job_probs)<=1:
            continue
        job_probs = sorted(job_probs, key=lambda x: -x)
        if job_probs[0]-job_probs[1]<=delta:
            interesting_steps.append(i)
    return interesting_steps

def update_finished_nodes(unfinished_nodes, finished_nodes):
    for node in unfinished_nodes:
        if node.num_finished_tasks==node.num_tasks:
            finished_nodes.add(node)
    unfinished_nodes.difference_update(finished_nodes)

# Func to keep track of finished jobs
def update_finished_job_dags(unfinished_job_dags, finished_job_dags):
    for job_dag in unfinished_job_dags:
        if job_dag.completed:
            finished_job_dags.add(job_dag)
    unfinished_job_dags.difference_update(finished_job_dags)

# Scheduling (witout attacks, but with trace recording)
def run_with_trace_recording(env, agent, steps=-1, verbosity=2, noLocality=False):
    # stuff to keep track of trace
    obss = []
    trace_node_probs = []
    trace_job_probs = []
    trace_job_names = []
    # sets for keeping track of finished jobs and nodes
    unfinished_job_dags = set([job_dag for job_dag in env.job_dags])
    unfinished_nodes = set()
    for job_dag in unfinished_job_dags:
        unfinished_nodes.update(set(job_dag.nodes))
    finished_job_dags = set()
    finished_nodes = set()
    # run
    done = False
    step = 0
    obs = env.observe()
    nodeInputs, jobInputs, \
        jobDags, sourceJob, numSourceExec, \
        frontierNodes, _, \
        _, _, execMap, actionMap = agent.translate_state(obs, noLocality)

    if steps == 0:
        return [obs], \
            trace_node_probs, \
            trace_job_probs, \
            trace_job_names

    while not done:
        # check whether any node or job DAG is done
        update_finished_nodes(unfinished_nodes, finished_nodes)
        update_finished_job_dags(unfinished_job_dags, finished_job_dags)
        # get action
        node, use_exec, feed_dict, \
          node_probs = agent.get_action(obs, rich_output=True, sparse=True, noLocality=noLocality)
        # update trace
        job_dags = obs[0]
        if (node is not None) and len(job_dags)>1:
            #nodeInputs, jobInputs, \
            #   jobDags, sourceJob, numSourceExec, \
            #   frontierNodes, _, \
            #   _, _, execMap, actionMap = agent.translate_state(obs, noLocality=noLocality)
            #print("source job", sourceJob)
            #print("num source execs:", numSourceExec)
            #print(jobInputs)
            #print(nodeInputs[:,:3])
            obss.append(obs)
            node_probs = np.squeeze(node_probs)
            trace_node_probs.append(node_probs)
            summ_mats = feed_dict[agent.gsn.summ_mats_sparse[0]]
            summ_mats = tf.SparseTensorValue(indices=summ_mats.indices, \
                                             values=[1.]*len(summ_mats.indices), \
                                              dense_shape=summ_mats.dense_shape) # convert to tf.float32
            job_probs = agent.sess.run(tf.sparse.sparse_dense_matmul(summ_mats, \
                                                                     node_probs[:, np.newaxis]))
            job_probs = [jp for jp in np.squeeze(job_probs)]
            trace_job_probs.append( job_probs )
            trace_job_names.append([job_dag.name for job_dag in env.job_dags])
        # update env
        if verbosity >= 2:
            print(f"\rStep {step}", end="")
        if steps != -1 and step == steps:
            break
        obs, _, done = env.step(node, use_exec)
        step += 1

    print("")
    # updated completed nodes and job DAGs
    update_finished_nodes(unfinished_nodes, finished_nodes)
    update_finished_job_dags(unfinished_job_dags, finished_job_dags)
    # done
    return obss, \
        trace_node_probs, \
        trace_job_probs, \
        trace_job_names

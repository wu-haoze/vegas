"""
Used to generate a bunch of jobs with one node/stage and a small
number of tasks each.
"""

from spark_env.job_dag import *
from spark_env.node import *
from spark_env.task import *
from spark_env.job_generator import pre_process_task_duration, \
    recursive_find_descendant

MIN_TASKS = 6
MAX_TASKS = 10
N_TRAIN_EXECUTORS = 50
TASK_MEAN_TIME_FIRST_WAVE = 4000
#TASK_MEAN_TIME_REST_WAVE = 4000
TASK_MEAN_TIME_REST_WAVE = 2000
TASK_TIME_STD = 50

def sample_task_time(is_first_wave, np_random):
    if is_first_wave:
        mean = TASK_MEAN_TIME_FIRST_WAVE
    else:
        mean = TASK_MEAN_TIME_REST_WAVE
    time = int(TASK_TIME_STD*np_random.randn() + mean)
    return max(1, time)

def get_small_job(np_random, wall_time, jid):
    """
    Returns a small job with a single stage and a random number of tasks between 
    MIN_TASKS and MAX_TASKS.
    """
    adj_mat = np.zeros((1, 1))

    # task durations
    n_tasks = np_random.randint(MIN_TASKS, MAX_TASKS)
    task_duration = {"first_wave": {N_TRAIN_EXECUTORS: [sample_task_time(True, np_random)]},
                     "rest_wave": {N_TRAIN_EXECUTORS: [sample_task_time(False, np_random) for _ in range(n_tasks-1)]},
                     "fresh_durations": {N_TRAIN_EXECUTORS: [sample_task_time(True, np_random)]}}
    task_durations = [task_duration]

    # the rest is taken from job_generator.load_job
    job_dag = init_job(adj_mat, task_durations, np_random, wall_time, jid)
    return job_dag


def init_job(adj_mat, task_durations, np_random, wall_time, jid):
    """
    taken from job_generator.load_job
    """
    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert adj_mat.shape[0] == len(task_durations)

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])
        print(f"Number of tasks for node {n} of job {jid}: {num_tasks}")

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l])
        print(f"Rough task duration for node {n} of job {jid}: {rough_duration:0.2f}")

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, tasks, task_duration, wall_time, np_random)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, adj_mat, jid)

    return job_dag


def generate_jobs(np_random, timeline, wall_time, n_small_dags=51):

    job_dags = OrderedSet()
    t = 0

    # add small dags
    for jid in range(n_small_dags):
        job_dag = get_small_job(np_random, wall_time, f"small_job_{jid}")
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    # No streaming dags for now...
    # for _ in range(args.num_stream_dags):
    #     # poisson process
    #     t += int(np_random.exponential(args.stream_interval))
    #     # uniform distribution
    #     query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
    #     query_idx = str(np_random.randint(args.tpch_num) + 1)
    #     # generate job
    #     job_dag = load_job(
    #         args.job_folder, query_size, query_idx, wall_time, np_random)
    #     # push into timeline
    #     job_dag.start_time = t
    #     timeline.push(t, job_dag)

    return job_dags

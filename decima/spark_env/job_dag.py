import networkx as nx
import numpy as np
from collections import OrderedDict
from utils import OrderedSet
from param import *
from spark_env.node import Node, NodeDuration
from spark_env.task import Task
from spark_env.action_map import compute_act_map

class JobDAG(object):
    def __init__(self, nodes, adj_mat, name):
        # nodes: list of N nodes
        # adj_mat: N by N 0-1 adjacency matrix, e_ij = 1 -> edge from i to j
        assert len(nodes) == adj_mat.shape[0]
        assert adj_mat.shape[0] == adj_mat.shape[1]

        self.name = name

        self.nodes = nodes
        self.adj_mat = adj_mat

        self.num_nodes = len(self.nodes)
        self.num_nodes_done = 0

        # set of executors currently running on the job
        self.executors = OrderedSet()

        # the computation graph needs to be a DAG
        assert is_dag(self.num_nodes, self.adj_mat)

        # get the set of schedule nodes
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)

        # assign job dag to node
        self.assign_job_dag_to_node()

        # dag is arrived
        self.arrived = False

        # dag is completed
        self.completed = False

        # dag start ime
        self.start_time = None

        # dag completion time
        self.completion_time = np.inf

        # map a executor number to an interval
        self.executor_interval_map = \
            self.get_executor_interval_map()

    def assign_job_dag_to_node(self):
        for node in self.nodes:
            node.job_dag = self

    def get_executor_interval_map(self):
        executor_interval_map = {}
        entry_pt = 0

        # get the left most map
        for e in range(args.executor_data_point[0] + 1):
            executor_interval_map[e] = \
                (args.executor_data_point[0],
                 args.executor_data_point[0])

        # get the center map
        for i in range(len(args.executor_data_point) - 1):
            for e in range(args.executor_data_point[i] + 1,
                            args.executor_data_point[i + 1]):
                executor_interval_map[e] = \
                    (args.executor_data_point[i],
                     args.executor_data_point[i + 1])
            # at the data point
            e = args.executor_data_point[i + 1]
            executor_interval_map[e] = \
                (args.executor_data_point[i + 1],
                 args.executor_data_point[i + 1])

        # get the residual map
        if args.exec_cap > args.executor_data_point[-1]:
            for e in range(args.executor_data_point[-1] + 1,
                            args.exec_cap + 1):
                executor_interval_map[e] = \
                    (args.executor_data_point[-1],
                     args.executor_data_point[-1])

        return executor_interval_map

    def get_nodes_duration(self):
        # Warning: this is slow O(num_nodes * num_tasks)
        # get the duration over all nodes
        duration = 0
        for node in self.nodes:
            duration += node.get_node_duration()
        return duration

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.num_nodes_done = 0
        self.executors = OrderedSet()
        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)
        self.arrived = False
        self.completed = False
        self.completion_time = np.inf

    def update_frontier_nodes(self, node):
        frontier_nodes_changed = False
        for child in node.child_nodes:
            if child.is_schedulable():
                if child.idx not in self.frontier_nodes:
                    self.frontier_nodes.add(child)
                    frontier_nodes_changed = True
        return frontier_nodes_changed
    
    def update_nodes(self, env, features_orig, features_new, normalize_features=True):
        """
        Added by Mahmood.
        The goal of this method is to enable adding new nodes to
        the job and updating the features of the existing nodes.
        """
        
        if normalize_features:
            features_new = features_new.copy()
            features_new[:, 3] = features_new[:, 3]*100000.
            features_new[:, 4] = features_new[:, 4]*200
            features_orig = features_orig.copy()
            features_orig[:, 3] = features_orig[:, 3]*100000.
            features_orig[:, 4] = features_orig[:, 4]*200
        
        # add new nodes
        wt = self.nodes[0].wall_time
        npr = self.nodes[0].np_random
        for i in range(len(features_new)):
            # create node, per features
            fs = features_new[i]
            node_idx = len(self.nodes)
            node_n_tasks = int(fs[4])
            node_task_duration = fs[3]/node_n_tasks
            tasks = [Task(j, node_task_duration, wt) for j in range(node_n_tasks)]
            task_duration = {"first_wave": {args.exec_cap: [node_task_duration]},
                             "rest_wave": {args.exec_cap: [node_task_duration for _ in range(node_n_tasks-1)]},
                             "fresh_durations": {args.exec_cap: [node_task_duration]}}
            node = Node(node_idx, \
                        tasks, \
                        task_duration, \
                        wt, \
                        npr)
            assert( np.isclose(node.num_tasks*node.tasks[-1].duration, \
                               fs[3], \
                               rtol=1e-04, \
                               atol=1e-06) ), \
                f'Error: {node.num_tasks*node.tasks[-1].duration} != {fs[3]}'
            assert( np.isclose(node.num_tasks, \
                               fs[4], \
                               rtol=1e-04, \
                               atol=1e-06) ), \
                f'Error: {node.num_tasks} != {fs[4]}'
            self.nodes.append(node)
            self.num_nodes += 1
            # update node's parent
            node.job_dag = self
            # update frontier nodes
            self.frontier_nodes.add(node)
            # update env's exec_commit
            env.exec_commit.commit[node] = OrderedDict()
            env.exec_commit.node_commit[node] = 0
            env.exec_commit.backward_map[node] = set()
            # update env's moving_executors
            env.moving_executors.node_track[node] = set()
            # update env's action_map
            env.action_map = compute_act_map(env.job_dags)
            
        # update adjacency matrix
        right_pad = np.zeros((len(features_orig), len(features_new)))
        self.adj_mat = np.concatenate([self.adj_mat, right_pad], axis=1)
        bottom_pad = np.zeros((len(features_new), len(features_orig)+len(features_new)))
        self.adj_mat = np.concatenate([self.adj_mat, bottom_pad], axis=0)
        
        # update other states?
        print('Need to update existing/old nodes and potentially other states')


def merge_job_dags(job_dags):
    # merge all DAGs into a general big DAG
    # this function will modify the original data structure
    # 1. take nodes from the natural order
    # 2. wire the parent and children across DAGs
    # 3. reconstruct adj_mat by properly connecting
    # the new edges among individual adj_mats

    total_num_nodes = sum([d.num_nodes for d in job_dags])
    nodes = []
    adj_mat = np.zeros([total_num_nodes, total_num_nodes])

    base = 0  # for figuring out new node index
    leaf_nodes = []  # leaf nodes in the current job_dag

    for job_dag in job_dags:

        num_nodes = job_dag.num_nodes

        for n in job_dag.nodes:
            n.idx += base
            nodes.append(n)

        # update the adj matrix
        adj_mat[base : base + num_nodes, \
            base : base + num_nodes] = job_dag.adj_mat

        # fundamental assumption of spark --
        # every job ends with a single final stage
        if base != 0:  # at least second job
            for i in range(num_nodes):
                if np.sum(job_dag.adj_mat[:, i]) == 0:
                    assert len(job_dag.nodes[i].parent_nodes) == 0
                    adj_mat[base - 1, base + i] = 1

        # store a set of new root nodes
        root_nodes = []
        for n in job_dag.nodes:
            if len(n.parent_nodes) == 0:
                root_nodes.append(n)

        # connect the root nodes with leaf nodes
        for root_node in root_nodes:
            for leaf_node in leaf_nodes:
                leaf_node.child_nodes.append(root_node)
                root_node.parent_nodes.append(leaf_node)

        # store a set of new leaf nodes
        leaf_nodes = []
        for n in job_dag.nodes:
            if len(n.child_nodes) == 0:
                leaf_nodes.append(n)

        # update base
        base += num_nodes

    assert len(nodes) == adj_mat.shape[0]

    merged_job_dag = JobDAG(nodes, adj_mat)

    return merged_job_dag


class JobDAGDuration(object):
    # A light-weighted extra storage for job_dag duration

    def __init__(self, job_dag):
        self.job_dag = job_dag

        self.node_durations = \
            {node: NodeDuration(node) for node in self.job_dag.nodes}

        for node in self.job_dag.nodes:
            # initialize descendant nodes duration
            self.node_durations[node].descendant_work = \
                np.sum([self.node_durations[n].duration \
                        for n in node.descendant_nodes])
            # initialize descendant nodes task duration
            self.node_durations[node].descendant_cp = \
                np.sum([n.tasks[0].duration \
                        for n in node.descendant_nodes])

        self.job_dag_duration = \
            np.sum([self.node_durations[node].duration \
                    for node in self.job_dag.nodes])

        self.nodes_done = {}

    def update_duration(self):
        work_done = 0
        for node in self.job_dag.nodes:
            if node not in self.nodes_done and node.tasks_all_done:
                work_done += self.node_durations[node].duration
                self.nodes_done[node] = node
        self.job_dag_duration -= work_done


def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)

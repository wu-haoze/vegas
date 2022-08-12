import numpy as np
import tensorflow as tf
import copy
import subprocess as sub
from os.path import isfile
from tempfile import NamedTemporaryFile
import os

from spark_env import env as orig_env
from deterministic_actor_agent import ActorAgent
from msg_passing_path import get_unfinished_nodes_summ_mat

from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from maraboupy import MarabouNetwork
from verification_util import load_benchmark_env,run_with_trace_recording,close_competition, load_multi_step_benchmark_env, close_node_competition
import itertools

def parseVec(net):
    return np.array(eval(net.readline()[:-1]))

class DecimaNetwork(MarabouNetwork.MarabouNetwork):
    """Constructs a Demica  network object

    Args:
        filename (str): Path to the .tf file
        env: Environment object
        params: Parameters
        verbosityLevel: The verbosity level

    Returns:
        :class:`~maraboupy.Marabou.marabouNetworkONNX.marabouNetworkONNX`
    """
    def __init__(self, filename, params, steps=1, checkNodeScore=True,
                 checkJobExecScore=True, verbosityLevel=2, agent=None,
                 multiStep=False, noLocality=False, closeNodeCompetition=False):
        super().__init__()

        self.clearNetwork()

        self.networkName = filename

        self.checkNodeScore = checkNodeScore
        self.checkJobExecScore = checkJobExecScore
        self.verbosityLevel = verbosityLevel
        self.multiStep = multiStep
        self.noLocality = noLocality

        # Magic numbers
        self.actorHiddenDimensions = [32,16,8]
        self.slope = 0.2 # Leaky ReLU slope

        self.getParameters(params)

        env = self.getEnv()
        self.getNetworkWeights()

        if agent == None:
            self.initiateAgent()
        else:
            self.agent = agent
            self.sess = agent.sess

        self.steps = steps
        self.lastStep = steps
        self.observations, traceNodeProbs, traceJobProbs, \
            _ = run_with_trace_recording(env, self.agent, steps, verbosityLevel)

        self.allSteps = len(traceJobProbs)

        if closeNodeCompetition:
            self.candidateSteps = close_node_competition(traceNodeProbs)
        else:
            self.candidateSteps = close_competition(traceJobProbs)

        self.log(f"{len(self.candidateSteps)} candidate steps.")
        self.log(f"candidate steps: {self.candidateSteps}")

    def getEnv(self):
        if self.useBenchmark > 0:
            np.random.seed(self.seed)
            tf.random.set_random_seed(self.seed)
            if self.multiStep:
                env = load_multi_step_benchmark_env(self.numJobs, self.numExecutors, self.rep)
            else:
                env = load_benchmark_env(self.numJobs, self.numExecutors, self.rep)
        else:
            env = orig_env.Environment()
            # set the seed
            env.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_random_seed(self.seed)
            env.reset()
        return env

    def clearNetwork(self):
        self.clear()

        self.numNodes = None
        self.numUnfinishedJobs = None

        # network topology
        self.gcnMats = None
        self.gcnMasks = None

        # Various mappings
        self.nodeToNodeIdx = dict()
        self.nodeIdxToNode = dict()
        self.jobToJobIdx = dict()
        self.jobIdxToJob = dict()
        # a mapping from job index to the node indices in the job
        self.jobToNodes = dict()
        # a reverse mapping from node index to job index
        self.nodeIdxToJobIdx = dict()
        # a mapping from node index to its feature variables
        self.nodeIdxToFeatures = dict()
        # a mapping from job index to its feature variables
        self.jobIdxToFeatures = dict()
        # a mapping from executor level to its variable
        self.executorLevels = dict()
        # a mapping from node index to its latent embedding
        self.nodeIdxToEncoding = dict()
        # a mapping from node index to its output after message passing
        self.nodeIdxToGCNOutput = dict()
        self.jobIdxToJobSummary = dict()
        # a mapping from job index to its pre-softmax score variable
        self.nodeIdxToScore = dict()
        # a mapping from (job, #executor) pair to its pre-softmax score variable
        self.jobExecToScore = dict()
        # Global summary variables
        self.globalSummary = []

        # Nodes and (job-exec) pairs participating in the classification
        self.validNodes = []
        self.validJobExecs = []

        # Initial states
        self.nodeInputs = None
        self.jobInputs = None

        self.abstractNodeIdx = -1

        self.disjunctToNodeIdx = dict()

    def getParameters(self, params):
        # Basic parameters
        self.numExecutors = params.exec_cap
        self.numJobs = params.num_init_dags
        self.rep = params.rep
        self.seed= params.seed
        self.useBenchmark = params.use_benchmark
        assert(params.num_stream_dags==0)
        assert(params.test_schemes == ["learn"])

        # network topology
        self.numNodeFeatures = params.node_input_dim
        self.messagePassingDepth = params.max_depth
        self.numJobFeatures = params.job_input_dim
        self.outputDimensions = params.output_dim
        self.hiddenDimensions = params.hid_dims

    def initiateAgent(self):
        self.log("Initiating agent...")
        self.sess = tf.Session()
        executor_levels = range(1, self.numExecutors + 1) # Look out for off-by-1
        self.agent = ActorAgent(
            self.sess, self.numNodeFeatures, self.numJobFeatures,
            self.hiddenDimensions, self.outputDimensions,
            self.messagePassingDepth, executor_levels)
        self.log("Initiating agent - done")

    def setInitialState(self, step):
        self.log(f"Setting initial state to step {step}...")
        self.clearNetwork()

        if self.lastStep != step:
            env = self.getEnv()
            self.observations, _, _, _ = run_with_trace_recording(env,
                                                                  self.agent,
                                                                  step, 2,
                                                                  self.noLocality)
            self.lastStep = step
        obs = self.observations[step]

        self.setInitialStateFromObs(obs)
        self.log(f"Setting initial state to step {step} - done")
        return

    def setInitialStateFromObs(self, obs):
        self.nodeInputs, self.jobInputs, \
        jobDags, sourceJob, numSourceExec, \
        frontierNodes, _, \
        _, _, execMap, actionMap = self.agent.translate_state(obs,
                                                              noLocality=self.noLocality)

        self.numNodes = len(self.nodeInputs)
        self.log(f"Number of nodes is {self.numNodes}", 1)

        self.log("Registering job and node indices")
        nodeIdx = 0
        jobIdx = 0
        self.jobToJobIdx.clear()
        self.jobIdxToJob.clear()
        self.nodeToNodeIdx.clear()
        self.nodeIdxToNode.clear()
        self.nodeIdxToJobIdx.clear()
        self.jobToNodes.clear()

        for jobDag in jobDags:
            self.jobToJobIdx[jobDag] = jobIdx
            self.jobIdxToJob[jobIdx] = jobDag
            nodeIndices = []
            for node in jobDag.nodes:
                self.nodeToNodeIdx[node] = nodeIdx
                self.nodeIdxToNode[nodeIdx] = node
                nodeIndices.append(nodeIdx)
                self.nodeIdxToJobIdx[nodeIdx] = jobIdx
                nodeIdx += 1
            self.jobToNodes[jobIdx] = nodeIndices
            jobIdx += 1
            with open("node_edge.csv", 'a') as out_file:
                out_file.write("{},{}\n".format(len(jobDag.nodes), np.sum(jobDag.adj_mat)))

        self.log("Registering job and node indices - done")

        self.gcnMats, self.gcnMasks, _, runningDagMat, _ = \
        self.agent.postman.get_msg_path(jobDags)

        self.unfinishedJobs = np.where(self.sess.run(
            tf.sparse.to_dense(tf.sparse.reorder(runningDagMat))).flatten() == 1)[0]
        self.unfinishedJobs.sort()

        self.numUnfinishedJobs = len(self.unfinishedJobs)

        nodeValidMask, jobValidMask = \
            self.agent.get_valid_masks(jobDags, frontierNodes,
                                       sourceJob, numSourceExec,
                                       execMap, actionMap)

        self.validNodes = np.where(nodeValidMask[0] == 1)[0]
        jobs, execs = np.where(jobValidMask.reshape
                               (self.numUnfinishedJobs,
                                self.numExecutors) == 1)
        self.validJobExecs = list(zip(jobs, execs))

        self.log("Get message passing matrices...")
        gcnMats = self.sess.run([tf.sparse.to_dense(tf.sparse.reorder(gcnMat))
                                 for gcnMat in self.gcnMats])
        self.gcnMats = gcnMats
        self.log("Get message passing matrices - done")
        return

    def getWeights(self, netFile, archs):
        netWeights = []
        netBias = []
        for i in range(len(archs))[1:]:
            weights = np.array(eval(netFile.readline()[:-1]))
            bias = np.array(eval(netFile.readline()[:-1]))
            assert(weights.shape == (archs[i], archs[i - 1]))
            assert(bias.shape == (archs[i],))
            netWeights.append(weights)
            netBias.append(bias)
        return netWeights, netBias

    def getNetworkWeights(self):
        self.log("Getting weights...")
        netFile = open(self.networkName, 'r')
        self.prepW, self.prepB = self.getWeights(netFile,
                                                 [self.numNodeFeatures] +
                                                 self.hiddenDimensions +
                                                 [self.outputDimensions])
        self.procW, self.procB = self.getWeights(netFile,
                                                 [self.outputDimensions] +
                                                 self.hiddenDimensions +
                                                 [self.outputDimensions])
        self.aggW, self.aggB = self.getWeights(netFile,
                                               [self.outputDimensions] +
                                               self.hiddenDimensions +
                                               [self.outputDimensions])
        self.dagW, self.dagB = self.getWeights(netFile,
                                               [self.numNodeFeatures +
                                                self.outputDimensions] +
                                               self.hiddenDimensions +
                                               [self.outputDimensions])
        self.globalW, self.globalB = self.getWeights(netFile,
                                                     [self.outputDimensions] +
                                                     self.hiddenDimensions +
                                                     [self.outputDimensions])

        self.nodePredW, self.nodePredB = \
            self.getWeights(netFile, [self.numNodeFeatures +
                                      3 * self.outputDimensions] +
                            self.actorHiddenDimensions +
                            [1])
        self.jobExecPredW, self.jobExecPredB = \
            self.getWeights(netFile, [self.numJobFeatures +
                                      2 * self.outputDimensions + 1] +
                            self.actorHiddenDimensions +
                            [1])
        self.log("Getting weights - done")
        return

    def addFullyConnectedLayer(self, weights, bias, activated=True,
                               inputs=None, outputs=None):
        shape0, shape1 = weights.shape[0], weights.shape[1]
        assert(shape0 == bias.shape[0])
        if inputs == None:
            inputs = [self.getNewVariable() for i in range(shape1)]
        if activated:
            preActivation = [self.getNewVariable() for i in range(shape0)]
            assert(len(preActivation) == shape0)
        if outputs == None:
            outputs = [self.getNewVariable() for i in range(shape0)]
        assert(len(inputs) == shape1)
        assert(len(outputs) == shape0)

        for i in range(shape0):
            e = MarabouUtils.Equation()
            for j in range(shape1):
                e.addAddend(weights[i][j], inputs[j])
            # Put output variable as the last addend last
            if activated:
                e.addAddend(-1, preActivation[i])
            else:
                e.addAddend(-1, outputs[i])
            e.setScalar(-bias[i])
            self.addEquation(e)

            if activated:
                self.addLeakyRelu(preActivation[i], outputs[i], self.slope)
        return outputs

    def createDecimaNetwork(self, abstraction=-1):
        self.log("Creating Decima network...")
        self.createInputVariables()
        self.encodeNodePreparation()
        self.encodeMessagePassing()
        self.encodeDAGandGlobalSummary()
        self.outputVars = []
        if self.checkNodeScore:
            self.encodeNodeClassification()
            self.outputVars.append(np.array([self.nodeIdxToScore[x]
                                             for x in self.validNodes]))
            if abstraction != -1:
                self.encodeNodeScoreAbstraction(abstraction)

        if self.checkJobExecScore:
            self.encodeJobClassification()
            self.outputVars.append(np.array([self.jobExecToScore[x]
                                             for x in self.validJobExecs]))

        self.log("Creating Decima network - done")
        self.log(f"{self.numVars} variables, {len(self.equList)} equations, "
                 f"{len(self.leakyReluList)} leakyReLUs")

    def createInputVariables(self):
        self.log("Creating input variables...")
        # node features
        for i in range(self.numNodes):
            self.nodeIdxToFeatures[i] = [self.getNewVariable()
                                         for j in range(self.numNodeFeatures)]
        # job features
        for i in range(self.numUnfinishedJobs):
            self.jobIdxToFeatures[i] = [self.getNewVariable()
                                         for j in range(self.numJobFeatures)]
        # Executor levels
        for i in range(self.numExecutors):
            self.executorLevels[i] = self.getNewVariable()
            self.setLowerBound(self.executorLevels[i], (i + 1)/50)
            self.setUpperBound(self.executorLevels[i], (i + 1)/50)

        nodeVars = []
        for i in range(self.numNodes):
            for var in self.nodeIdxToFeatures[i]:
                nodeVars.append(var)
        jobVars = []
        for i in range(self.numUnfinishedJobs):
            for var in self.jobIdxToFeatures[i]:
                jobVars.append(var)

        self.inputVars = [np.array(nodeVars),
                          np.array(jobVars),
                          np.array([self.executorLevels[i]
                                    for i in range(self.numExecutors)])]
        self.log("Creating input variables - done")

    def reindexNodeVariables(self, finishedJob):
        # node features
        numNodesInFinishedJob = len(self.jobToNodes[finishedJob])

        for i in range(self.numNodes):
            if self.nodeIdxToJobIdx[i] == finishedJob:
                del self.nodeIdxToFeatures[i]
                del self.nodeIdxToEncoding[i]
                del self.nodeIdxToGCNOutput[i]
            elif self.nodeIdxToJobIdx[i] > finishedJob:
                assert((i - numNodesInFinishedJob) not in self.nodeIdxToFeatures)
                assert((i - numNodesInFinishedJob) not in self.nodeIdxToEncoding)
                assert((i - numNodesInFinishedJob) not in self.nodeIdxToGCNOutput)

                self.nodeIdxToFeatures[i - numNodesInFinishedJob] = self.nodeIdxToFeatures[i]
                del self.nodeIdxToFeatures[i]

                self.nodeIdxToEncoding[i - numNodesInFinishedJob] = self.nodeIdxToEncoding[i]
                del self.nodeIdxToEncoding[i]

                self.nodeIdxToGCNOutput[i - numNodesInFinishedJob] = self.nodeIdxToGCNOutput[i]
                del self.nodeIdxToGCNOutput[i]

        for i in range(self.numUnfinishedJobs):
            if i == finishedJob:
                del self.jobIdxToJobSummary[i]
            elif i > finishedJob:
                assert((i - 1) not in self.jobIdxToJobSummary)

                self.jobIdxToJobSummary[i - 1] = self.jobIdxToJobSummary[i]
                del self.jobIdxToJobSummary[i]

    def encodeNodePreparation(self):
        self.log("Encoding node preparation...")
        for i in range(self.numNodes):
            inputs = self.nodeIdxToFeatures[i]
            for j in range(len(self.prepW)):
                outputs = self.addFullyConnectedLayer(self.prepW[j],
                                                      self.prepB[j],
                                                      activated=True,
                                                      inputs=inputs)
                inputs = outputs
            assert(len(outputs) == self.outputDimensions)
            self.nodeIdxToEncoding[i] = outputs
        self.log("Encoding node preparation - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)

    def encodeMessagePassing(self):
        self.log("Encoding message passing...")
        currentNodeIdxToMessageVars = copy.copy(self.nodeIdxToEncoding)
        for d in range(self.messagePassingDepth):
            childrenComputed = dict() # map from child node u to f(e_u)
            parents = np.where(self.gcnMasks[d].flatten() == 1)[0]
            for parent in parents:
                children = np.where(self.gcnMats[d][parent, :].flatten() == 1)[0]
                childrenMessages = []
                # Process the features on the node
                for child in set(children):
                    if child not in childrenComputed:
                        # current message of the children
                        inputs = currentNodeIdxToMessageVars[child]
                        for j in range(len(self.procW)):
                            outputs = self.addFullyConnectedLayer(self.procW[j],
                                                                  self.procB[j],
                                                                  activated=True,
                                                                  inputs=inputs)
                            inputs = outputs
                        assert(len(outputs) == self.outputDimensions)
                        childrenComputed[child] = outputs
                    childrenMessages.append(childrenComputed[child])

                messageVars = self.computeMessage(parent, childrenMessages,
                                                  currentNodeIdxToMessageVars)
                assert(len(messageVars) == self.outputDimensions)
                currentNodeIdxToMessageVars[parent] = messageVars

        self.nodeIdxToGCNOutput = currentNodeIdxToMessageVars
        self.log("Encoding message passing - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)

    def encodeMessagePassingForAdditionalJob(self, jobIndex):
        self.log("Encoding message passing...")
        currentNodeIdxToMessageVars = copy.copy(self.nodeIdxToEncoding)
        nodesInJob = self.jobToNodes[jobIndex]

        for nodeIdx in self.nodeIdxToGCNOutput:
            if nodeIdx in self.jobToNodes[jobIndex]:
                self.nodeIdxToGCNOutput[nodeIdx] = self.nodeIdxToEncoding[nodeIdx]

        for d in range(self.messagePassingDepth):
            childrenComputed = dict() # map from child node u to f(e_u)
            parents = np.where(self.gcnMasks[d].flatten() == 1)[0]
            for parent in parents:
                if parent in nodesInJob:
                    children = np.where(self.gcnMats[d][parent, :].flatten() == 1)[0]
                    childrenMessages = []
                    # Process the features on the node
                    for child in set(children):
                        if child not in childrenComputed:
                            # current message of the children
                            inputs = currentNodeIdxToMessageVars[child]
                            for j in range(len(self.procW)):
                                outputs = self.addFullyConnectedLayer(self.procW[j],
                                                                      self.procB[j],
                                                                      activated=True,
                                                                      inputs=inputs)
                                inputs = outputs
                            assert(len(outputs) == self.outputDimensions)
                            childrenComputed[child] = outputs
                        childrenMessages.append(childrenComputed[child])

                    messageVars = self.computeMessage(parent, childrenMessages,
                                                      currentNodeIdxToMessageVars)
                    assert(len(messageVars) == self.outputDimensions)
                    currentNodeIdxToMessageVars[parent] = messageVars

        self.nodeIdxToGCNOutput = currentNodeIdxToMessageVars
        self.log("Encoding message passing - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)

    def computeMessage(self, parent, childrenMessages,
                       currentNodeIdxToMessageVars):
        # compute equation (1) in https://arxiv.org/pdf/1810.01963.pdf
        #we have [f(e_c1), f(e_c2), ...] first compute the sum of them
        self.log(f"Computing message for node {parent}...", level=2)
        sumVars = [self.getNewVariable() for i in range(self.outputDimensions)]
        for i, sumVar in enumerate(sumVars):
            e = MarabouUtils.Equation()
            for childMessage in childrenMessages:
                e.addAddend(1, childMessage[i])
            e.addAddend(-1, sumVar)
            e.setScalar(0)
            self.addEquation(e)

        # Computing g(sumVars)
        inputs = sumVars
        for i in range(len(self.aggW)):
            outputs = self.addFullyConnectedLayer(self.aggW[i], self.aggB[i],
                                                  activated=True, inputs=inputs)
            inputs = outputs
        assert(len(outputs) == self.outputDimensions)

        # Finally e_parent = g(sumVars) + xParent
        xParent = currentNodeIdxToMessageVars[parent]
        assert(len(xParent) == self.outputDimensions)

        messageVars = [self.getNewVariable()
                       for i in range(self.outputDimensions)]
        for i, msgVar in enumerate(messageVars):
            e = MarabouUtils.Equation()
            e.addAddend(1, outputs[i])
            e.addAddend(1, xParent[i])
            e.addAddend(-1, msgVar)
            e.setScalar(0)
            self.addEquation(e)
        self.log(f"Computing message for node {parent} - done", 2)
        return messageVars

    def encodeDAGandGlobalSummary(self):
        self.log("Encoding DAG and global summary...")

        nodeIdxToDagOuts = dict()
        for i in range(self.numNodes):
            inputs = self.nodeIdxToFeatures[i] + self.nodeIdxToGCNOutput[i]
            assert(len(inputs) == self.numNodeFeatures + self.outputDimensions)
            for j in range(len(self.dagW)):
                outputs = self.addFullyConnectedLayer(self.dagW[j], self.dagB[j],
                                                      activated=True,
                                                      inputs=inputs)
                inputs = outputs
            nodeIdxToDagOuts[i] = outputs

        # DAG summary
        for i in range(self.numUnfinishedJobs):
            jobSummaryVars = [self.getNewVariable()
                              for q in range(self.outputDimensions)]
            # Take the sum of the dag outs for nodes in the same job
            for j, jobSummaryVar in enumerate(jobSummaryVars):
                e = MarabouUtils.Equation()
                for nodeIdx in self.jobToNodes[i]:
                    e.addAddend(1, nodeIdxToDagOuts[nodeIdx][j])
                e.addAddend(-1, jobSummaryVar)
                e.setScalar(0)
                self.addEquation(e)

            self.jobIdxToJobSummary[i] = jobSummaryVars

        self.encodeGlobalSummary()

        self.log("Encoding DAG and global summary - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)

    def encodeGlobalSummary(self):
        # Global summary
        globalSummaryInputs = [self.getNewVariable()
                                  for i in range(self.outputDimensions)]
        for i, globalSummaryInput in enumerate(globalSummaryInputs):
            e = MarabouUtils.Equation()
            for jobIdx in self.unfinishedJobs:
                e.addAddend(1, self.jobIdxToJobSummary[jobIdx][i])
            e.addAddend(-1, globalSummaryInput)
            e.setScalar(0)
            self.addEquation(e)
        inputs = globalSummaryInputs
        for j in range(len(self.globalW)):
            outputs = self.addFullyConnectedLayer(self.globalW[j],
                                                  self.globalB[j],
                                                  activated=True,
                                                  inputs=inputs)
            inputs = outputs
        self.globalSummary = outputs


    def encodeAdditionalDAGandGlobalSummary(self, jobIdx):
        self.log("Encoding additional DAG and global summary...")

        nodesInJob = self.jobToNodes[jobIdx]

        nodeIdxToDagOuts = dict()
        for i in range(self.numNodes):
            if i in nodesInJob:
                inputs = self.nodeIdxToFeatures[i] + self.nodeIdxToGCNOutput[i]
                assert(len(inputs) == self.numNodeFeatures + self.outputDimensions)
                for j in range(len(self.dagW)):
                    outputs = self.addFullyConnectedLayer(self.dagW[j], self.dagB[j],
                                                          activated=True,
                                                          inputs=inputs)
                    inputs = outputs
                nodeIdxToDagOuts[i] = outputs

        # DAG summary
        jobSummaryVars = [self.getNewVariable()
                          for i in range(self.outputDimensions)]
        # Take the sum of the dag outs for nodes in the same job
        for j, jobSummaryVar in enumerate(jobSummaryVars):
            e = MarabouUtils.Equation()
            for nodeIdx in self.jobToNodes[jobIdx]:
                e.addAddend(1, nodeIdxToDagOuts[nodeIdx][j])
            e.addAddend(-1, jobSummaryVar)
            e.setScalar(0)
            self.addEquation(e)

        self.jobIdxToJobSummary[jobIdx] = jobSummaryVars

        self.encodeGlobalSummary()

        self.log("Encoding additional DAG and global summary - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)

    def encodeNodeClassification(self):
        self.log("Encoding node classification...")
        self.log(f"Number of valid nodes: {len(self.validNodes)}")
        for nodeIdx in self.validNodes:
            inputs = (self.nodeIdxToFeatures[nodeIdx] +
                      self.nodeIdxToGCNOutput[nodeIdx] +
                      self.jobIdxToJobSummary[self.nodeIdxToJobIdx[nodeIdx]] +
                      self.globalSummary)
            for j in range(len(self.nodePredW)):
                activated = True if j != len(self.nodePredW) - 1 else False
                outputs = self.addFullyConnectedLayer(self.nodePredW[j],
                                                      self.nodePredB[j],
                                                      activated=activated,
                                                      inputs=inputs)
                inputs = outputs
            assert(len(outputs) == 1)
            self.nodeIdxToScore[nodeIdx] = outputs[0]
        self.log("Encoding node classification - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)
        return

    def encodeJobClassification(self):
        self.log("Encoding job-exec classification...")
        self.log(f"Number of valid job-exec pairs: {len(self.validJobExecs)}")
        for (jobIdx, execIdx) in self.validJobExecs:
            inputs = (self.jobIdxToFeatures[jobIdx] +
                      self.jobIdxToJobSummary[jobIdx] +
                      self.globalSummary + [self.executorLevels[execIdx]])

            for j in range(len(self.jobExecPredW)):
                activated = True if j != len(self.jobExecPredW) - 1 else False
                outputs = self.addFullyConnectedLayer(self.jobExecPredW[j],
                                                      self.jobExecPredB[j],
                                                      activated=activated,
                                                      inputs=inputs)
                inputs = outputs
            assert(len(outputs) == 1)
            self.jobExecToScore[(jobIdx, execIdx)] = outputs[0]
        self.log("Encoding job-exec classification - done")
        self.log(f"Number of LeakyReLU added so far: {len(self.leakyReluList)}", 2)
        return

    def encodeNodeScoreAbstraction(self, jobIndex):
        self.log("Adding an aux node for job {}".format(jobIndex))

        self.abstractNodeIdx = self.numNodes
        self.nodeIdxToFeatures[self.abstractNodeIdx] = [self.getNewVariable()
                                                        for i in
                                                        range(self.numNodeFeatures)]
        self.nodeIdxToGCNOutput[self.abstractNodeIdx] = [self.getNewVariable()
                                                        for i in
                                                        range(self.outputDimensions)]
        for var in self.nodeIdxToFeatures[self.abstractNodeIdx]:
            self.setLowerBound(var, 0)
            self.setUpperBound(var, 0)
        for var in self.nodeIdxToGCNOutput[self.abstractNodeIdx]:
            self.setLowerBound(var, 0)
            self.setUpperBound(var, 0)

        inputs = (self.nodeIdxToFeatures[self.abstractNodeIdx] +
                  self.nodeIdxToGCNOutput[self.abstractNodeIdx] +
                  self.jobIdxToJobSummary[jobIndex] +
                  self.globalSummary)
        for j in range(len(self.nodePredW)):
            activated = True if j != len(self.nodePredW) - 1 else False
            outputs = self.addFullyConnectedLayer(self.nodePredW[j],
                                                  self.nodePredB[j],
                                                  activated=activated,
                                                  inputs=inputs)
            inputs = outputs
        assert(len(outputs) == 1)

        self.inputVars.append(np.array(self.nodeIdxToFeatures[self.abstractNodeIdx]))
        self.inputVars.append(np.array(self.nodeIdxToGCNOutput[self.abstractNodeIdx]))
        self.nodeIdxToScore[self.abstractNodeIdx] = outputs[0]
        self.log("Adding an aux node - done")

    def evaluateWithMarabou(self, nodeInputs=None, jobInputs=None, options=None):
        if nodeInputs is None:
            nodeInputs = self.nodeInputs
        if jobInputs is None:
            jobInputs = self.jobInputs

        assert(nodeInputs.shape == (self.numNodes, self.numNodeFeatures))
        assert(jobInputs.shape == (self.numUnfinishedJobs, self.numJobFeatures))

        for i, nodeFeatures in enumerate(nodeInputs):
            for j, feature in enumerate(nodeFeatures):
                self.setLowerBound(self.nodeIdxToFeatures[i][j], feature)
                self.setUpperBound(self.nodeIdxToFeatures[i][j], feature)

        for i, jobFeatures in enumerate(jobInputs):
            for j, feature in enumerate(jobFeatures):
                self.setLowerBound(self.jobIdxToFeatures[i][j], feature)
                self.setUpperBound(self.jobIdxToFeatures[i][j], feature)

        self.log("Getting Marabou query...", 2)
        ipq = self.getMarabouQuery()
        self.log("Getting Marabou query - done", 2)

        if options == None:
            options = MarabouCore.Options()
            options._verbosity = 0
            options._solveWithMILP = True
        outputDict, _ = MarabouCore.solve(ipq, options)

        # When the query is UNSAT an empty dictionary is returned
        if outputDict == {}:
            return None

        outputValues = []
        for outputs in self.outputVars:
            outputValue = outputs.reshape(-1).astype(np.float64)
            for i in range(len(outputValue)):
                outputValue[i] = outputDict[outputValue[i]]
            outputValues.append(outputValue.reshape(outputs.shape))
        return outputValues

    def dumpValidNodeFeatures(self, nodeInputs=None, jobInputs=None, options=None):
        if nodeInputs is None:
            nodeInputs = self.nodeInputs
        if jobInputs is None:
            jobInputs = self.jobInputs

        assert(nodeInputs.shape == (self.numNodes, self.numNodeFeatures))
        assert(jobInputs.shape == (self.numUnfinishedJobs, self.numJobFeatures))

        for i, nodeFeatures in enumerate(nodeInputs):
            for j, feature in enumerate(nodeFeatures):
                self.setLowerBound(self.nodeIdxToFeatures[i][j], feature)
                self.setUpperBound(self.nodeIdxToFeatures[i][j], feature)

        for i, jobFeatures in enumerate(jobInputs):
            for j, feature in enumerate(jobFeatures):
                self.setLowerBound(self.jobIdxToFeatures[i][j], feature)
                self.setUpperBound(self.jobIdxToFeatures[i][j], feature)

        self.log("Getting Marabou query...", 2)
        ipq = self.getMarabouQuery()
        self.log("Getting Marabou query - done", 2)

        if options == None:
            options = MarabouCore.Options()
            options._verbosity = 0
            options._solveWithMILP = True
        outputDict, _ = MarabouCore.solve(ipq, options)

        # When the query is UNSAT an empty dictionary is returned
        if outputDict == {}:
            return None

        for job in self.unfinishedJobs:
            print(f"Job {job}")
            for ind in self.jobToNodes[job]:
                if ind not in self.validNodes:
                    continue
                print(f"\tnode {ind}:")
                feats = "\tfeatures: "
                for var in self.nodeIdxToFeatures[ind]:
                    feats += f"{outputDict[var]:.2f} "
                print(feats)
                encoding = "\tpost msg passing encoding: "
                for var in self.nodeIdxToGCNOutput[ind]:
                    encoding += f"{outputDict[var]:.2f} "
                print(encoding)
                print(f"\tscore: {outputDict[self.nodeIdxToScore[ind]]}")

    def evaluateWithoutMarabou(self, nodeInputs=None, jobInputs=None):
        if nodeInputs is None:
            nodeInputs = self.nodeInputs
        if jobInputs is None:
            jobInputs = self.jobInputs

        obs = self.observations[self.lastStep]

        # Copied from deterministic_actor_agent.py:invoke_model()
        _, _, \
            job_dags, source_job, num_source_exec, \
            frontier_nodes, executor_limits, \
            exec_commit, moving_executors, \
            exec_map, action_map = self.agent.translate_state(obs)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, \
            running_dags_mat, job_dags_changed = \
            self.agent.postman.get_msg_path(job_dags)

        # get node and job valid masks
        node_valid_mask, job_valid_mask = \
            self.agent.get_valid_masks(job_dags, frontier_nodes,
                source_job, num_source_exec, exec_map, action_map)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        # invoke learning model
        node_act_logits, job_act_logits = \
        self.agent.predict(nodeInputs, jobInputs,
                           node_valid_mask, job_valid_mask,
                           gcn_mats, gcn_masks, summ_mats,
                           running_dags_mat, dag_summ_backward_map, sparse=False, logits=True)

        node_act_probs_nonzero = [node_act_logits[0][i] for i in self.validNodes]
        job_act_probs_nonzero = [job_act_logits[0][i] for i in self.validJobExecs]

        assert(len(node_act_probs_nonzero) == len(self.validNodes))
        assert(len(job_act_probs_nonzero) == len(self.validJobExecs))

        return [node_act_probs_nonzero, job_act_probs_nonzero]

    def log(self, msg, level=2):
        if level <= self.verbosityLevel:
            print(msg)

    def solve(self, verbose=True, dumpBounds=False, backwardBoundTightening=True,
              deleteQueryFile=True, timeout=0, queryFileName=None,
              solveAllDisjuncts=False, countTimeout=False):
        """Function to solve query represented by this network

        Args:
            verbose (bool): If true, print out solution after solve finishes
            options (:class:`~maraboupy.MarabouCore.Options`): Object for specifying Marabou options, defaults to None

        Returns:
            (tuple): tuple containing:
                - vals (Dict[int, float]): Empty dictionary if UNSAT, otherwise a dictionary of SATisfying values for variables
                - stats (:class:`~maraboupy.MarabouCore.Statistics`): A Statistics object to how Marabou performed
        """
        for i, nodeFeatures in enumerate(self.nodeInputs):
            for j, feature in enumerate(nodeFeatures):
                var = self.nodeIdxToFeatures[i][j]
                if var not in self.lowerBounds:
                    self.setLowerBound(var, feature)
                if var not in self.upperBounds:
                    self.setUpperBound(var, feature)

        for i, jobFeatures in enumerate(self.jobInputs):
            for j, feature in enumerate(jobFeatures):
                var = self.jobIdxToFeatures[i][j]
                if var not in self.lowerBounds:
                    self.setLowerBound(var, feature)
                if var not in self.upperBounds:
                    self.setUpperBound(var, feature)

        if queryFileName == None:
            f = NamedTemporaryFile(delete=False)
            queryFileName = f.name
            print(f"Query file name: {queryFileName}")
            self.saveQuery(queryFileName)
        else:
            self.saveQuery(queryFileName)
            return None, None

        summaryFileName = queryFileName + ".summary"

        command = (f"/home/gagandeep/andrew/robust-decima/resources/Marabou/build/Marabou "
                   f"--input-query {queryFileName} "
                   f"--milp --summary-file={summaryFileName} --num-workers=32 "
                   f"--timeout={timeout}")
        if dumpBounds:
            self.log("Dumping bounds...", 2)
            command += " --dump-bounds"
        if not verbose:
            command += " --verbosity=0"
        if backwardBoundTightening:
            self.log("Performing backward analysis...", 2)
            command += " --backward --converge"
        if solveAllDisjuncts:
            self.log("Solving all disjuncts...", 2)
            command += " --solve-all-disjuncts"
        sub.run(command.split())

        if solveAllDisjuncts:
            if not countTimeout:
                notInfeasible = []
                if isfile(summaryFileName):
                    resultStrings = open(summaryFileName, 'r').readlines()
                    assert(len(resultStrings) == len(self.disjunctToNodeIdx))
                    for resultString in resultStrings:
                        disjunct, result = resultString.strip().split()
                        if result != "unsat":
                            notInfeasible.append(self.disjunctToNodeIdx[int(disjunct)])
                    os.remove(summaryFileName)
                else:
                    self.log("No result file detected!", 0)
                    assert(False)
                if deleteQueryFile:
                    os.remove(queryFileName)
                return notInfeasible
            else:
                notInfeasible = []
                sat = []
                unknown = []
                if isfile(summaryFileName):
                    resultStrings = open(summaryFileName, 'r').readlines()
                    assert(len(resultStrings) == len(self.disjunctToNodeIdx))
                    for resultString in resultStrings:
                        disjunct, result = resultString.strip().split()
                        if result != "unsat":
                            notInfeasible.append(self.disjunctToNodeIdx[int(disjunct)])
                            if result == "sat":
                                sat.append(self.disjunctToNodeIdx[int(disjunct)])
                            else:
                                unknown.append(self.disjunctToNodeIdx[int(disjunct)])
                    os.remove(summaryFileName)
                else:
                    self.log("No result file detected!", 0)
                    assert(False)
                if deleteQueryFile:
                    os.remove(queryFileName)
                return notInfeasible, sat, unknown

        else:
            result = "UNKNOWN"
            vals = dict()
            if isfile(summaryFileName):
                resultStrings = open(summaryFileName, 'r').readlines()
                result = resultStrings[0].strip().split()[0]
                if result == 'sat':
                    for i, line in enumerate(resultStrings[1:]):
                        vals[i] = float(line.split(",")[1])
                    for ind in self.validNodes:
                        print("Pre-softmax node score:")
                        print(f"\tnode {ind} in job {self.nodeIdxToJobIdx[ind]}:"
                              f"{vals[self.nodeIdxToScore[ind]]}")
                os.remove(summaryFileName)
            if deleteQueryFile:
                os.remove(queryFileName)
            return result, vals

    def addConstraintNodeMax(self, nodeIdx):
        # Add the constraints that specifies that a given node is the largest output
        if nodeIdx not in self.validNodes:
            print("The node is not in the frontier!")
            print("Frontier nodes: ", self.validNodes )
            exit(0)
        self.log(f"Adding constraint that node {nodeIdx} is max...", 1)
        print("Frontier nodes: ", self.validNodes )
        for otherNodeIdx in self.validNodes:
            if ( otherNodeIdx != nodeIdx ):
                self.addInequality( [self.nodeIdxToScore[nodeIdx],
                                     self.nodeIdxToScore[otherNodeIdx]], [-1,1], 0)
        return


    def computeVolume(self, nodeIndices):
        volume = 1
        for i in range(self.numNodeFeatures):
            minLb = 100000
            maxUb = -100000
            for nodeIdx in nodeIndices:
                var = self.nodeIdxToFeatures[nodeIdx][i]
                nodeInput = self.nodeInputs[nodeIdx].reshape(1,self.numNodeFeatures)
                ub = self.upperBounds[var] if var in self.upperBounds else nodeInput[0][i]
                lb = self.lowerBounds[var] if var in self.lowerBounds else nodeInput[0][i]
                if ub > maxUb:
                    maxUb = ub
                if lb < minLb:
                    minLb = lb
            if minLb != maxUb:
                volume *= (maxUb - minLb)
        return volume

    def addConstraintJobMax(self, jobIdx):
        # Add the constraints that specifies one of the node in a job has the max output
        self.log(f"\nAdding constraint that a node in job {jobIdx} is max...", 1)
        self.disjunctToNodeIdx = dict()

        disjuncts = []
        if self.abstractNodeIdx != -1:
            self.log(f"Add abstract node disjunct...", 1)
            self.log(f"Score var {self.nodeIdxToScore[self.abstractNodeIdx]}", 2)
            conjuncts = []
            for otherNodeIdx in self.validNodes:
                if otherNodeIdx != self.abstractNodeIdx:
                    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
                    eq.setScalar(0)
                    eq.addAddend(-1, self.nodeIdxToScore[self.abstractNodeIdx])
                    eq.addAddend(1, self.nodeIdxToScore[otherNodeIdx])
                    conjuncts.append(eq)
            disjuncts.append(conjuncts)
            self.log(f"Add abstract node disjunct - done")

        # Compute box width:
        nodes = []
        for nodeIdx in self.jobToNodes[jobIdx]:
                if nodeIdx in self.validNodes:
                    nodes.append(nodeIdx)

        numNodes = len(nodes)

        removalOrderFromAbstraction = []

        while len(removalOrderFromAbstraction) != numNodes:
            # iteratively find the box that will decrease the volume the most.
            minVolume = 10000000
            minVolumeNodeIdx = -1
            self.log(f"current nodes: {nodes}",2)
            for node in nodes:
                oneOff = copy.copy(nodes)
                oneOff.remove(node)
                newVolume = self.computeVolume(oneOff)
                assert(newVolume <= self.computeVolume(nodes))
                if newVolume < minVolume:
                    minVolume = newVolume
                    minVolumeNodeIdx = node
            assert(minVolumeNodeIdx in nodes and minVolumeNodeIdx not in removalOrderFromAbstraction)
            removalOrderFromAbstraction.append(minVolumeNodeIdx)
            nodes.remove(minVolumeNodeIdx)

        dindex = 0
        for i, nodeIdx in enumerate(removalOrderFromAbstraction):
            self.log(f"Input volume with nodeIdx {nodeIdx} is {self.computeVolume(removalOrderFromAbstraction[i:])}",2)
            self.disjunctToNodeIdx[dindex] = nodeIdx
            conjuncts = []
            for otherNodeIdx in self.validNodes:
                if otherNodeIdx != nodeIdx:
                    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
                    eq.setScalar(0)
                    eq.addAddend(-1, self.nodeIdxToScore[nodeIdx])
                    eq.addAddend(1, self.nodeIdxToScore[otherNodeIdx])
                    conjuncts.append(eq)
            disjuncts.append(conjuncts)
            dindex += 1
        self.addDisjunctionConstraint(disjuncts)

        if self.abstractNodeIdx != -1:
            self.log("Adding equivalence class...")
            for i in range(self.numNodeFeatures):
                equiv = []
                equiv.append(self.nodeIdxToFeatures[self.abstractNodeIdx][i])
                for nodeIdx in removalOrderFromAbstraction:
                    equiv.append(self.nodeIdxToFeatures[nodeIdx][i])
                self.addEquivalence(equiv)
            for i in range(self.outputDimensions):
                equiv = []
                equiv.append(self.nodeIdxToGCNOutput[self.abstractNodeIdx][i])
                for nodeIdx in removalOrderFromAbstraction:
                    equiv.append(self.nodeIdxToGCNOutput[nodeIdx][i])
                self.addEquivalence(equiv)
            self.log("Adding equivalence class - done")
        return

    def addConstraintNodeNotMax(self, nodeIdx):
        # Add the constraints that specifies a given node is not the max output
        if nodeIdx not in self.validNodes:
            print("The node is not in the frontier!")
            print("Frontier nodes: ", self.validNodes )
            exit(0)
        self.log(f"Adding constraint that node {nodeIdx} is not max...", 1)
        print("Frontier nodes: ", self.validNodes )
        disjuncts = []
        for otherNodeIdx in self.validNodes:
            if ( otherNodeIdx != nodeIdx ):
                eq = MarabouCore.Equation(MarabouCore.Equation.LE)
                eq.setScalar(0)
                eq.addAddend(1, self.nodeIdxToScore[nodeIdx])
                eq.addAddend(-1, self.nodeIdxToScore[otherNodeIdx])
                disjuncts.append([eq])
            self.addDisjunctionConstraint(disjuncts)
        return

    def setMaxJobExecForVerification(self, jobIdx, numExecutors):
        # Add the constraints that specifies whether a given job-exec pair will be classified
        assert(False)
        return

    def getTopNodeIndex(self):
        nodeScores, jobExecScores = self.evaluateWithoutMarabou()
        topNodeIdx = self.validNodes[np.argmax(nodeScores)]
        self.log(f"Top node index is {topNodeIdx}", 1)
        return topNodeIdx

    def getTopJobIndex(self):
        return self.nodeIdxToJobIdx[self.getTopNodeIndex()]

    def getTopNodeIndexInSecondJob(self):
        nodeScores, jobExecScores = self.evaluateWithoutMarabou()
        sortedNodeIndices = np.array(nodeScores).argsort().tolist()
        sortedNodeIndices.reverse()
        topJobIdx = self.nodeIdxToJobIdx[self.validNodes[sortedNodeIndices[0]]]
        for ind in sortedNodeIndices:
            thisJobIdx = self.nodeIdxToJobIdx[self.validNodes[ind]]
            if  thisJobIdx!= topJobIdx:
                topNodeIdx = self.validNodes[ind]
                self.log(f"Top node index in second job is {topNodeIdx}", 1)
                return topNodeIdx
        self.log(f"Top node index in second job not found", 0)
        exit()

    def getSecondJobIndex(self):
        return self.nodeIdxToJobIdx[self.getTopNodeIndexInSecondJob()]

    def setNodeFeatureBounds(self, nodeIdx, lower=None, upper=None, featureIdx=1):
        var = self.nodeIdxToFeatures[nodeIdx][featureIdx]
        if lower != None:
            self.log(f"setting lower bound of x{var} to {lower}", 2)
            self.setLowerBound(var, lower)
        if upper != None:
            self.log(f"setting upper bound of x{var} to {upper}", 2)
            self.setUpperBound(var, upper)
        return

    def setPerTaskDurationLowerBound(self, nodeIdx, lowerBound):
        durationVar = self.nodeIdxToFeatures[nodeIdx][3]
        taskVar = self.nodeIdxToFeatures[nodeIdx][4]
        self.addInequality([taskVar, durationVar], [lowerBound, -1], 0)

    def addNodeToJob(self, jobIdx, isFrontier=True):
        newNodeInput = copy.copy(self.nodeInputs[self.jobToNodes[jobIdx][0]]).reshape(1,self.numNodeFeatures)
        self.nodeInputs = np.concatenate([self.nodeInputs, newNodeInput], axis=0)

        newNodeIdx = self.numNodes
        self.numNodes += 1

        self.nodeIdxToJobIdx[newNodeIdx] = jobIdx
        self.jobToNodes[jobIdx].append(newNodeIdx)

        for d in range(self.messagePassingDepth):
            rightPad = np.zeros((self.numNodes - 1, 1))
            self.gcnMats[d] = np.concatenate((self.gcnMats[d], rightPad), axis=1)
            bottomPad = np.zeros((1, self.numNodes))
            self.gcnMats[d] = np.concatenate((self.gcnMats[d], bottomPad), axis=0)

            bottomPad = np.zeros((1, 1))
            self.gcnMasks[d] = np.concatenate((self.gcnMasks[d],
                                               bottomPad), axis=0)

        if isFrontier:
            rightPad = np.zeros(1)
            rightPad[0] = newNodeIdx
            self.validNodes = np.concatenate((self.validNodes, rightPad), axis=0)

        return newNodeIdx

    # Grow the proof transfer graph.
    def growProofTransferGraph(self, nextStateObs, lastScheduledNodeIndex):

        self.log("Growing proof transfer graph...", 2)
        self.addConstraintNodeMax(lastScheduledNodeIndex)

        lastNumUnfinishedJobs = self.numUnfinishedJobs
        jobIndex = self.nodeIdxToJobIdx[lastScheduledNodeIndex]

        nodeInputs, jobInputs, \
        jobDags, sourceJob, numSourceExec, \
        frontierNodes, _, \
        _, _, execMap, actionMap = self.agent.translate_state(nextStateObs,
                                                              noLocality=self.noLocality)

        gcnMats, gcnMasks, _, runningDagMat, _ = \
                                                 self.agent.postman.get_msg_path(jobDags)

        unfinishedJobs = np.where(self.sess.run(
            tf.sparse.to_dense(tf.sparse.reorder(runningDagMat))).flatten() == 1)[0]
        numUnfinishedJobs = len(unfinishedJobs)
        self.log(f"Unfinished jobs before: {lastNumUnfinishedJobs}, "
                 f"unfinished jobs now: {numUnfinishedJobs}", 2)


        jobFinished = (numUnfinishedJobs < lastNumUnfinishedJobs)

        # Case 1: the job is not complete.
        if not jobFinished:
            self.log("No jobs completed at this step...", 2)

            self.setInitialStateFromObs(nextStateObs)
            assert(lastNumUnfinishedJobs == self.numUnfinishedJobs)

            self.encodeMessagePassingForAdditionalJob(jobIndex)
            self.encodeAdditionalDAGandGlobalSummary(jobIndex)
            self.outputVars = []
            if self.checkNodeScore:
                self.encodeNodeClassification()
                self.outputVars.append(np.array([self.nodeIdxToScore[x]
                                                 for x in self.validNodes]))
            if self.checkJobExecScore:
                assert(False)

        else: # Case 2: the job is completed.
            self.log(f"Job {jobIndex} completed...", 2)
            self.reindexNodeVariables(jobIndex)

            self.setInitialStateFromObs(nextStateObs)

            self.encodeGlobalSummary()
            self.outputVars = []
            if self.checkNodeScore:
                self.encodeNodeClassification()
                self.outputVars.append(np.array([self.nodeIdxToScore[x]
                                                 for x in self.validNodes]))

            if self.checkJobExecScore:
                assert(False)

        self.log("Growing proof transfer graph - done", 2)

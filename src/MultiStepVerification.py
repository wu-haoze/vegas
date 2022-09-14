from src_verify.DecimaNetwork import DecimaNetwork
from src_verify.ScheduleTemplate import ScheduleTemplate, ComparisonResult
from verification_util import load_multi_step_benchmark_env,run_with_trace_recording
from spark_env import env as orig_env
from deterministic_actor_agent import ActorAgent
from msg_passing_path import get_unfinished_nodes_summ_mat

from copy import deepcopy
import numpy as np
import tensorflow as tf

class StackEntry:
    def __init__(self):
        self.currentNodeIndex = -1
        self.nodeIndicesToExplore = []

class MultiStepVerification:
    def __init__(self, template, params, verbosityLevel=2):
        self.template = template
        self.getParameters(params)
        self.verbosityLevel = verbosityLevel
        self.noLocality = True

        self.stack = []
        self.sequenceToNodeIndices = dict()

        self.decimaNetwork = DecimaNetwork("gcn.tf", params, steps=0,
                                           checkNodeScore=True,
                                           checkJobExecScore=False,
                                           verbosityLevel=2, multiStep=True,
                                           noLocality = self.noLocality)
        self.agent = self.decimaNetwork.agent

        obs = self.getObservationFromSchedule([])
        self.decimaNetwork.setInitialStateFromObs(obs)

        self.initialJobToNodes = deepcopy(self.decimaNetwork.jobToNodes)
        self.initialNodeIdxToJobIdx \
            = deepcopy(self.decimaNetwork.nodeIdxToJobIdx)
        self.initialUnfinishedJobs = deepcopy(self.decimaNetwork.unfinishedJobs)
        print("Initial job index to node index: ", self.initialJobToNodes)
        print("Initial unfinished jobs: ", self.initialUnfinishedJobs)

        self.perturbations = []

    def getTrace(self, step=-1):
        trace = []
        env = self.getEnv()
        _, traceNodeProbs, _, _ = run_with_trace_recording(env,
                                                           self.agent,
                                                           step,
                                                           2,
                                                           noLocality=self.noLocality)
        scheduleSoFar = []
        for nodeProbs in traceNodeProbs:
            nodeIdx = np.argmax(nodeProbs)
            reindexed = self.getIndexInInitialState(scheduleSoFar,nodeIdx)
            scheduleSoFar.append(reindexed)
            trace.append((reindexed, self.initialNodeIdxToJobIdx[reindexed]))
        return trace

    def getParameters(self, params):
        # Basic parameters
        self.numExecutors = params.exec_cap
        self.numJobs = params.num_init_dags
        self.rep = params.rep
        self.seed= params.seed
        self.initialStep = params.steps
        self.useBenchmark = params.use_benchmark
        self.proofTransfer = params.proof_transfer

        assert(params.num_stream_dags==0)
        assert(params.test_schemes == ["learn"])

    def getEnv(self):
        if self.useBenchmark > 0:
            np.random.seed(self.seed)
            tf.random.set_random_seed(self.seed)
            env = load_multi_step_benchmark_env(self.numJobs, self.numExecutors, self.rep)
        else:
            env = orig_env.Environment()
            # set the seed
            env.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_random_seed(self.seed)
            env.reset()
        return env

    def addPerturbation(self, nodeIdx, lower=None, upper=None, featureIdx=1):
        self.perturbations.append([nodeIdx, lower, upper, featureIdx])
        return

    def encodePerturbations(self, currentSchedule):
        for perturbation in self.perturbations:
            nodeIdx, lower, upper, featureIdx = perturbation
            currentIdx = self.getIndexInCurrentState(currentSchedule, nodeIdx)
            if currentIdx != -1:
                self.decimaNetwork.setNodeFeatureBounds(currentIdx, lower,
                                                        upper, featureIdx)
        return

    def sortedSchedule(self, nodeIndices):
        return "-".join(list(map(str,sorted(nodeIndices))))

    def getIndexInCurrentState(self, currentSchedule, nodeIdx):
        self.log(f"Getting index for {nodeIdx} in current state "
                 f"with current schedule {currentSchedule}", 3)

        # Compute which jobs are done
        tempMap = deepcopy(self.initialJobToNodes)
        for currentNodeIndex in currentSchedule:
            jIdx = self.initialNodeIdxToJobIdx[currentNodeIndex]
            tempMap[jIdx].remove(currentNodeIndex)

        finishedJobs = []
        unfinishedJobs = []
        for jobIdx in tempMap:
            if len(tempMap[jobIdx]) == 0:
                finishedJobs.append(jobIdx)
            else:
                unfinishedJobs.append(jobIdx)
        finishedJobs = sorted(finishedJobs)
        unfinishedJobs = sorted(unfinishedJobs)

        thisJobIdx = self.initialNodeIdxToJobIdx[nodeIdx]
        if thisJobIdx in finishedJobs:
            return -1

        offset = 0
        for finishedJob in finishedJobs:
            if finishedJob < thisJobIdx:
                offset += len(self.initialJobToNodes[finishedJob])
        assert(offset <= nodeIdx)

        self.log(f"Index in current state is {nodeIdx - offset}", 3)
        return nodeIdx - offset

    def getJobIndexInCurrentState(self, jobIndex, trace, step):
        self.log(f"Getting index for {jobIndex} in current state "
                 f"with trace {trace}", 3)

        # Compute which jobs are done
        tempMap = deepcopy(self.initialJobToNodes)
        for currentNodeIndex, job in trace[:step]:
            jIdx = self.initialNodeIdxToJobIdx[currentNodeIndex]
            tempMap[jIdx].remove(currentNodeIndex)

        finishedJobs = []
        unfinishedJobs = []
        for jobIdx in tempMap:
            if len(tempMap[jobIdx]) == 0:
                finishedJobs.append(jobIdx)
            else:
                unfinishedJobs.append(jobIdx)
        finishedJobs = sorted(finishedJobs)
        unfinishedJobs = sorted(unfinishedJobs)

        offset = 0
        for job in finishedJobs:
            if job < jobIndex:
                offset += 1
        return jobIndex - offset

    def getIndexInInitialState(self, currentSchedule, nodeIdx):
        self.log(f"Getting index for {nodeIdx} in initial state "
                 f"with current schedule {currentSchedule}", 3)

        # Compute which jobs are done
        tempMap = deepcopy(self.initialJobToNodes)
        for currentNodeIndex in currentSchedule:
            jIdx = self.initialNodeIdxToJobIdx[currentNodeIndex]
            tempMap[jIdx].remove(currentNodeIndex)

        finishedJobs = []
        unfinishedJobs = []
        for jobIdx in tempMap:
            if len(tempMap[jobIdx]) == 0:
                finishedJobs.append(jobIdx)
            else:
                unfinishedJobs.append(jobIdx)
        finishedJobs = sorted(finishedJobs)
        unfinishedJobs = sorted(unfinishedJobs)

        numNodes = 0
        thisJob = -1
        for unFinishedJob in unfinishedJobs:
            numNodes += len(self.initialJobToNodes[unFinishedJob])
            if numNodes > nodeIdx:
                thisJob = unFinishedJob
                break
        assert(numNodes > nodeIdx)
        assert(thisJob >= 0)

        offset = 0
        for job in finishedJobs:
            if job < thisJob:
                offset += len(self.initialJobToNodes[job])

        self.log(f"Index in initial state is {nodeIdx + offset}", 3)
        return nodeIdx + offset

    def verify(self):
        currentSchedule = []
        while True:
            result, impliedNextNode = self.template.compareSchedule(currentSchedule)
            if result == ComparisonResult.MATCH:
                self.log("Current schedule: {currentSchedule} matches"
                         " the target template!", 1)
                return "unknown"
            elif result == ComparisonResult.NOT_MATCH:
                self.log(f"current schedule: {currentSchedule} inconsistent "
                         "with the target template! \nBacktracking...", 2)
                newSchedule = self.popFromStack()
                if len(newSchedule) == 0:
                    return "unsat"
                else:
                    currentSchedule = newSchedule
                    self.log(f"Current schedule updated to {currentSchedule}",
                             2)
                    continue
            else:
                assert(result == ComparisonResult.INCONCLUSIVE)
                self.log(f"Computing possible next steps from {currentSchedule}"
                         , 2)
                if impliedNextNode == None:
                    nodeIndices = self.getPossibleNodeIndices(currentSchedule)
                else:
                    nodeIndices = self.checkNodeIndexFeasibility(currentSchedule,
                                                                 impliedNextNode)
                self.log(f"Possible next actions are {nodeIndices}", 2)
                self.pushToStack(nodeIndices)
                currentSchedule += [self.stack[-1].currentNodeIndex]
                self.log(f"Current schedule updated to {currentSchedule}", 2)

    def getObservationFromSchedule(self, scheduleSoFar):
        env = self.getEnv()
        self.observations, _, _, _ = run_with_trace_recording(env,
                                                              self.agent,
                                                              self.initialStep,
                                                              2, noLocality=self.noLocality)
        obs = self.observations[-1]

        for i, nodeIdx in enumerate(scheduleSoFar):
            newNodeIdx = self.getIndexInCurrentState(scheduleSoFar[:i], nodeIdx)
            assert(newNodeIdx >= 0)
            node, use_exec = self.agent.get_action(obs, sparse=True, noLocality=self.noLocality)
            node = obs[7][newNodeIdx]
            obs, _, _ = env.step(node, use_exec)
        return obs

    def getEncodingFromSchedule(self, scheduleSoFar, jobIdx):
        env = self.getEnv()
        self.observations, _, _, _ = run_with_trace_recording(env,
                                                              self.agent,
                                                              self.initialStep,
                                                              2, noLocality=self.noLocality)
        obs = self.observations[-1]

        self.decimaNetwork.setInitialStateFromObs(obs)
        self.decimaNetwork.createDecimaNetwork()
        self.encodePerturbations([])

        for i, nodeIdx in enumerate(scheduleSoFar):
            newNodeIdx = self.getIndexInCurrentState(scheduleSoFar[:i], nodeIdx)
            assert(newNodeIdx >= 0)
            node, use_exec = self.agent.get_action(obs, sparse=True, noLocality=self.noLocality)
            node = obs[7][newNodeIdx]
            obs, _, _ = env.step(node, use_exec)

            self.decimaNetwork.growProofTransferGraph(obs, newNodeIdx)

    def getPossibleNodeIndices(self, scheduleSoFar):
        indices = []
        obs = self.getObservationFromSchedule(scheduleSoFar)
        self.decimaNetwork.setInitialStateFromObs(obs)

        unfinishedJobs = deepcopy(self.decimaNetwork.unfinishedJobs)
        self.log(f"Unfinished jobs: {unfinishedJobs}", 2)
        for jobIdx in [3]:
        #for jobIdx in unfinishedJobs:
            self.decimaNetwork.clear()
            if self.proofTransfer:
                self.getEncodingFromSchedule(scheduleSoFar, jobIdx)
            else:
                self.decimaNetwork.setInitialStateFromObs(obs)
                self.decimaNetwork.createDecimaNetwork(jobIdx)
                self.encodePerturbations(scheduleSoFar)

            self.decimaNetwork.addConstraintJobMax(jobIdx)
            nodeIndices = self.decimaNetwork.solve(deleteQueryFile=True,
                                                   solveAllDisjuncts=True,
                                                   verbose=False,
                                                   backwardBoundTightening=False,
                                                   timeout=120)
            self.log(f"feasible node indices before reindexing: {nodeIndices}",
                     2)
            reindexed = [self.getIndexInInitialState(scheduleSoFar, index) for
                         index in nodeIndices]
            self.log(f"feasible node indices after reindexing: {reindexed}", 2)
            indices += reindexed

        return indices

    def checkNodeIndexFeasibility(self, scheduleSoFar, nodeIdx):
        indices = []
        obs = self.getObservationFromSchedule(scheduleSoFar)
        self.decimaNetwork.clear()
        if self.proofTransfer:
            self.getEncodingFromSchedule(scheduleSoFar, -1)
        else:
            self.decimaNetwork.setInitialStateFromObs(obs)
            self.decimaNetwork.createDecimaNetwork(-1)
            self.encodePerturbations(scheduleSoFar)

        self.decimaNetwork.addConstraintNodeMax(self.getIndexInCurrentState(scheduleSoFar, nodeIdx))
        result, vals = self.decimaNetwork.solve(deleteQueryFile=True,
                                                solveAllDisjuncts=False,
                                                verbose=False,
                                                backwardBoundTightening=True)
        if result == "sat":
            return [nodeIdx]
        else:
            return [-1]

    def pushToStack(self, nodeIndices):
        entry = StackEntry()
        entry.currentNodeIndex = nodeIndices[0]
        entry.nodeIndicesToExplore = nodeIndices[1:]
        self.stack.append(entry)

    def popFromStack(self):
        if len(self.stack) == 0:
            return []
        lastEntry = self.stack[-1]
        while len(lastEntry.nodeIndicesToExplore) == 0:
            self.stack = self.stack[:-1]
            if len(self.stack) == 0:
                return []
            else:
                lastEntry = self.stack[-1]

        lastEntry.currentNodeIndex = lastEntry.nodeIndicesToExplore[0]
        lastEntry.nodeIndicesToExplore = lastEntry.nodeIndicesToExplore[1:]

        return [entry.currentNodeIndex for entry in self.stack]

    def log(self, message, verbosity=2):
        if verbosity <= self.verbosityLevel:
            print(message)

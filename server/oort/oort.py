import math
import logging
from collections import OrderedDict
from random import Random

import numpy as np


def _normalize_probabilities(weights):
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    arr = arr / total
    arr = arr / float(arr.sum())
    return arr


def create_training_selector(args):
    return _training_selector(args)


def create_testing_selector(data_distribution=None, client_info=None, model_size=None):
    return _testing_selector(data_distribution, client_info, model_size)


class _testing_selector:
    """Minimal testing selector wrapper retained for package compatibility."""

    def __init__(self, data_distribution=None, client_info=None, model_size=None):
        self.client_info = client_info
        self.model_size = model_size
        self.data_distribution = data_distribution
        if self.client_info:
            self.client_idx_list = list(range(len(client_info)))

    def update_client_info(self, client_ids, client_profile):
        return 0

    def _hoeffding_bound(self, dev_tolerance, capacity_range, total_num_clients, confidence=0.8):
        factor = (1.0 - 2 * total_num_clients / math.log(1 - math.pow(confidence, 1)) * (dev_tolerance / float(capacity_range)) ** 2)
        return (total_num_clients + 1.0) / factor

    def select_by_deviation(self, dev_target, range_of_capacity, total_num_clients, confidence=0.8, overcommit=1.1):
        return self._hoeffding_bound(dev_target, range_of_capacity, total_num_clients, confidence=confidence)

    def select_by_category(self, request_list, max_num_clients=None, greedy_heuristic=True):
        raise NotImplementedError("Testing selector category mode requires the full Oort LP utils stack.")


class _training_selector(object):
    """Oort's training selector."""

    def __init__(self, args, sample_seed=233):
        self.totalArms = OrderedDict()
        self.training_round = 0
        self.exploration = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.alpha = args.exploration_alpha
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.args = args
        self.round_threshold = args.round_threshold
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0
        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None
        np.random.seed(sample_seed)

    def register_client(self, clientId, feedbacks):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['reward'] = feedbacks['reward']
            self.totalArms[clientId]['duration'] = feedbacks['duration']
            self.totalArms[clientId]['time_stamp'] = self.training_round
            self.totalArms[clientId]['count'] = 0
            self.totalArms[clientId]['status'] = True
            self.unexplored.add(clientId)

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0
        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']
        return cntUtil / cnt

    def pacer(self):
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)
        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)
        self.successfulClients = set()

        if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:
            utilLastPacerRounds = sum(self.exploitUtilHistory[-2 * self.args.pacer_step:-self.args.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.args.pacer_step:])
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100.0, self.round_threshold + self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at %s to %s", self.training_round, self.round_threshold)
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at %s to %s", self.training_round, self.round_threshold)

    def update_client_util(self, clientId, feedbacks):
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = feedbacks['status']
        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)

    def get_blacklist(self):
        blacklist = []
        if self.args.blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True, key=lambda k: self.totalArms[k]['count'])
            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.args.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break
            predefined_max_len = self.args.blacklist_max_len * len(self.totalArms)
            if len(blacklist) > predefined_max_len:
                logging.warning("Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]
        return set(blacklist)

    def select_participant(self, num_of_clients, feasible_clients=None):
        viable_clients = feasible_clients if feasible_clients is not None else set([x for x in self.totalArms.keys() if self.totalArms[x]['status']])
        return self.getTopK(num_of_clients, self.training_round + 1, viable_clients)

    def update_duration(self, clientId, duration):
        if clientId in self.totalArms:
            self.totalArms[clientId]['duration'] = duration

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        self.training_round = cur_time
        self.blacklist = self.get_blacklist()
        self.pacer()

        scores = {}
        numOfExploited = 0
        exploreLen = 0
        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if int(x) in feasible_clients and int(x) not in self.blacklist]

        if self.round_threshold < 100.0 and client_list:
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            idx = min(int(len(sortedDuration) * self.round_threshold / 100.0), len(sortedDuration) - 1)
            self.round_prefer_duration = sortedDuration[idx]
        else:
            self.round_prefer_duration = float('inf')

        moving_reward = []
        staleness = []
        for clientId in orderedKeys:
            if self.totalArms[clientId]['reward'] > 0:
                creward = self.totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(cur_time - self.totalArms[clientId]['time_stamp'])

        if moving_reward:
            max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, self.args.clip_bound)
        else:
            max_reward, min_reward, range_reward, avg_reward, clip_value = 1.0, 0.0, 1.0, 0.0, 1.0
        if staleness:
            max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)
        else:
            max_staleness, min_staleness, range_staleness, avg_staleness = 1.0, 0.0, 1.0, 0.0

        for key in orderedKeys:
            if self.totalArms[key]['count'] > 0:
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1
                sc = (creward - min_reward) / float(range_reward) + math.sqrt(0.1 * math.log(max(cur_time, 1)) / max(1.0, self.totalArms[key]['time_stamp']))
                clientDuration = self.totalArms[key]['duration']
                if clientDuration > self.round_prefer_duration:
                    sc *= (float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.args.round_penalty
                scores[key] = sc

        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        clientLakes = list(scores.keys())
        exploitLen = min(int(numOfSamples * (1.0 - self.exploration)), len(clientLakes))
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)

        pickedClients = []
        augment_factor = 1
        if exploitLen > 0 and len(sortedClientUtil) > exploitLen:
            cut_off_util = scores[sortedClientUtil[exploitLen]] * self.args.cut_off_util
            candidate_clients = []
            for clientId in sortedClientUtil:
                if scores[clientId] < cut_off_util:
                    break
                candidate_clients.append(clientId)
            augment_factor = len(candidate_clients)
            probs = _normalize_probabilities([scores[key] for key in candidate_clients])
            pickedClients = list(np.random.choice(candidate_clients, exploitLen, p=probs, replace=False))
        self.exploitClients = pickedClients

        if len(self.unexplored) > 0:
            unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients]
            init_reward = {}
            for cl in unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']
                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= (float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.args.round_penalty
            exploreLen = min(len(unexplored), numOfSamples - len(pickedClients))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window * max(exploreLen, 1)), len(init_reward))]
            if exploreLen > 0 and pickedUnexploredClients:
                probs = _normalize_probabilities([init_reward[key] for key in pickedUnexploredClients])
                pickedUnexplored = list(np.random.choice(pickedUnexploredClients, exploreLen, p=probs, replace=False))
                self.exploreClients = pickedUnexplored
                pickedClients = pickedClients + pickedUnexplored
        else:
            self.exploration_min = 0.0
            self.exploration = 0.0

        while len(pickedClients) < numOfSamples and orderedKeys:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        return pickedClients

    def get_median_reward(self):
        feasible_rewards = [self.totalArms[x]['reward'] for x in list(self.totalArms.keys()) if int(x) not in self.blacklist]
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards) / float(len(feasible_rewards))
        return 0

    def get_client_reward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList) * clip_bound), len(aList) - 1)]
        _max = max(aList)
        _min = min(aList) * 0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList) / max(1e-4, float(len(aList)))
        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)

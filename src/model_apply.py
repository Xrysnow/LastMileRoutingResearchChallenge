from os import path
import sys, json, time
# from seq_distance import med_ratio
# from predictor import ref_zone_edges, get_history_ref
from timer import Timer
from reader2 import DataReader2
from concurrent import futures
import numpy as np

BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
# TRAINING_PATH = path.join(BASE_DIR, 'data/model_build_inputs/')
MODEL_PATH = path.join(BASE_DIR, 'data/model_build_outputs/')
APPLY_PATH = path.join(BASE_DIR, 'data/model_apply_inputs/new_')
OUTPUT_FILE = path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')


def get_history_ref(ins, content):
    znames = ins.zone_names()
    candidate = {}
    for name in znames:
        v = content.get(name)
        if v is not None:
            for i in v:
                if candidate.get(i) is None:
                    candidate[i] = 1
                else:
                    candidate[i] += 1
    result = []
    for k, v in candidate.items():
        if v > 0:
            result.append(k)
    return result


def ref_zone_edges(ins, refs, comm_ratio=0.0):
    nz = ins.zone_count()
    require = max(1, nz * comm_ratio)
    result = []
    for ref in refs:
        common = ins.common_zones(ref)
        if len(common) >= require:
            act = ref.actual_zone_sequence()
            common_in_ref = []
            for zname in common:
                common_in_ref.append(ref.zone_id(zname))
            act_in_common = [z in common_in_ref for z in act]
            edges = []
            for i in range(len(act_in_common) - 1):
                if act_in_common[i] and act_in_common[i + 1]:
                    # edge in ref
                    z1 = ins.zone_id(ref.zone_name(act[i]))
                    z2 = ins.zone_id(ref.zone_name(act[i + 1]))
                    edges.append([z1, z2])
            result.append([ref, edges])
    return result


def get_range(all_start, all_end, i, n):
    delta = (all_end - all_start) / n
    start = int(all_start + i * delta)
    end = int(all_start + (i + 1) * delta)
    if i == n - 1:
        end = all_end
    return start, end


def worker(p):
    i, n = p[0], p[1]
    reader = DataReader2(APPLY_PATH, training=False)
    reader_train = DataReader2(MODEL_PATH, training=True)
    total = reader.instance_count()
    start, end = get_range(0, total, i, n)
    print('worker {}/{}, range {}-{}/{}'.format(i, n, start, end, total))

    result = {}
    t = Timer()
    t1 = None
    zins = json.load(open(MODEL_PATH + 'zins.json', 'r'))
    for i in range(start, end):
        c = reader.instance(i)
        refs = get_history_ref(c, zins)
        refs = [reader_train.instance(ref) for ref in refs]
        ref_edges = ref_zone_edges(c, refs)
        zseq2 = c.simple_solve_zone_sequence2(ref_edges)
        path = c.simple_solve(zseq2)
        proposed = c.propose(path)
        result[reader.instance_id(i)] = {'proposed': proposed}
        if not t1:
            t1 = t.tic(log=False)
            print('worker {}: {:.2f}s'.format(i, t1))
    return result


def run():
    n = 12
    result = []
    param = []
    for i in range(n):
        param.append((i, n))
    t = Timer()
    with futures.ProcessPoolExecutor(max_workers=n) as executor:
        for fu in executor.map(worker, param):
            result.append(fu)
    print('{} workers finished in {:.2f}s'.format(n, t.tic(log=False)))
    final = {}
    for r in result:
        for k, v in r.items():
            final[k] = v
    with open(OUTPUT_FILE, 'w') as f:
        f.write(json.dumps(final))
    print("saved to '{}'".format(OUTPUT_FILE))


run()
print('apply finish')

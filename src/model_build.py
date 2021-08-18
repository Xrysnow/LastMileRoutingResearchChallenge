import os
import sys, json, time
import shutil
from reader2 import DataReader2
from timer import Timer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_PATH = os.path.join(BASE_DIR, 'data/model_build_inputs/')
MODEL_PATH = os.path.join(BASE_DIR, 'data/model_build_outputs/')


def run():
    print('reading...')
    reader = DataReader2(TRAINING_PATH, training=True)
    print('running...')
    total = reader.instance_count()
    t = Timer()
    print('dump zins...')
    zseq = []
    for i in range(total):
        ins = reader.instance(i)
        act = ins.actual_zone_sequence()
        act_name = [ins.zone_name(z) for z in act]
        zseq.append(act_name)
        if i % 500 == 0:
            print("read {}, {}s".format(i, t.tic(log=False)))
    zins = {}
    for i, seq in enumerate(zseq):
        for z in seq:
            v = zins.get(z)
            if v is None:
                zins[z] = [i]
            else:
                v.append(i)
    print('save zins...')
    with open(MODEL_PATH + 'zins.json', 'w+') as f:
        f.write(json.dumps(zins))
    print("saved to '{}'".format(MODEL_PATH + 'zins.json'))
    print('copy files...')
    with os.scandir(TRAINING_PATH) as entries:
        for entry in entries:
            if entry.is_file():
                src = TRAINING_PATH + entry.name
                target = MODEL_PATH + entry.name
                print('copy {}'.format(src))
                shutil.copy(src, target)


run()
print('build finish')

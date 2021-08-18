import math
import numpy as np


def med(a, b):
    '''Minimum Edit Distance'''
    aLen = len(a)
    bLen = len(b)
    if aLen == 0:
        return bLen
    if bLen == 0:
        return aLen
    v = np.zeros((aLen + 1, bLen + 1), dtype=np.int)
    v[0] = np.arange(bLen + 1)
    v[:, 0] = np.arange(aLen + 1)
    for i in range(1, aLen + 1):
        for j in range(1, bLen + 1):
            if a[i - 1] == b[j - 1]:
                v[i, j] = v[i - 1, j - 1]
            else:
                v[i, j] = 1 + min(v[i - 1, j - 1], v[i, j - 1], v[i - 1, j])
    return v[aLen, bLen]


def med_ratio(a, b):
    d = med(a, b)
    # return float(d) / (len(a) + len(b)) * 2
    return float(d) / max(len(a), len(b))


def hamming(a, b):
    aLen = len(a)
    bLen = len(b)
    if aLen != bLen:
        raise RuntimeError()
    if aLen == 0:
        return 0
    count = 0
    for i in range(aLen):
        if a[i] != b[i]:
            count += 1
    return float(count) / aLen


def jaro(first, second, winkler=True, winkler_ajustment=True, scaling=0.1):
    """
    :param first: word to calculate distance for
    :param second: word to calculate distance with
    :param winkler: same as winkler_ajustment
    :param winkler_ajustment: add an adjustment factor to the Jaro of the distance
    :param scaling: scaling factor for the Winkler adjustment
    :return: Jaro distance adjusted (or not)
    """
    # if not first or not second:
    #     raise RuntimeError()
    jaro = _score(first, second)
    if all([winkler, winkler_ajustment]):  # 0.1 as scaling factor
        cl = min(len(_get_prefix(first, second)), 4)
        return round((jaro + (scaling * cl * (1.0 - jaro))) * 100.0) / 100.0
    return jaro


def _get_matching_characters(first, second):
    common = []
    limit = math.floor(min(len(first), len(second)) / 2)
    if isinstance(second, str):
        for i, l in enumerate(first):
            left, right = int(max(0, i - limit)), int(min(i + limit + 1, len(second)))
            if l in second[left:right]:
                common.append(l)
                second = second[0:second.index(l)] + '*' + second[second.index(l) + 1:]
    else:
        for i, l in enumerate(first):
            left, right = int(max(0, i - limit)), int(min(i + limit + 1, len(second)))
            if l in second[left:right]:
                common.append(l)
                second = second[0:second.index(l)] + ['*'] + second[second.index(l) + 1:]
    return common


def _score(first, second):
    shorter, longer = first, second
    if len(first) > len(second):
        longer, shorter = shorter, longer
    m1 = _get_matching_characters(shorter, longer)
    m2 = _get_matching_characters(longer, shorter)
    if len(m1) == 0 or len(m2) == 0:
        return 0.0
    trans = math.floor(len([(f, s) for f, s in zip(m1, m2) if not f == s]) / 2.0)
    return (float(len(m1)) / len(shorter) +
            float(len(m2)) / len(longer) +
            float(len(m1) - trans) / len(m1)) / 3.0


def _get_diff_index(first, second):
    # if first == second:
    #     return -1
    if not first or not second:
        return 0
    max_len = min(len(first), len(second))
    for i in range(0, max_len):
        if not first[i] == second[i]:
            return i
    return max_len


def _get_prefix(first, second):
    if not first or not second:
        return []
    index = _get_diff_index(first, second)
    if index == -1:
        return first
    elif index == 0:
        return []
    else:
        return first[0:index]


if __name__ == '__main__':
    from timer import Timer


    def check(s1, s2):
        print(med_ratio(s1, s2), med_ratio(s2, s1))
        print(jaro(s1, s2), jaro(s2, s1))
        print('---')


    s1 = 'qwertyuiopasdfgh'
    s2 = 'zxcvb' + s1[:-5]
    s3 = s1[::-1]
    l1, l2, l3 = [], [], []
    for v in [[s1, l1], [s2, l2], [s3, l3]]:
        for s in v[0]:
            v[1].append(s)
    check(s1, s2)
    check(s1, s3)
    check(l1, l2)
    check(l1, l3)
    N = 1000
    t = Timer()
    for i in range(N):
        med_ratio(s1, s3)
    t.tic()
    for i in range(N):
        jaro(s1, s3)
    t.tic()

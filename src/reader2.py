import numpy as np
from reader import DataReader, InstanceData
import copy

ZERO2 = np.zeros((2, 2), dtype='float64')


def sqrtm(mat):
    e, p = np.linalg.eig(mat)
    e_0, e_1 = e[0], e[1]
    if e_0 <= 0.0 and e_1 <= 0.0:
        return ZERO2
    e0 = 0.0
    e1 = 0.0
    if e_0 > 0.0:
        e0 = np.sqrt(e[0])
    if e_1 > 0.0:
        e1 = np.sqrt(e[1])
    e_sqrt = np.array([[e0, 0], [0, e1]], dtype='float64')
    p_inv = np.linalg.inv(p)
    return np.linalg.multi_dot([p, e_sqrt, p_inv])


def normal_distance(mu1, s1, sq1, mu2, s2, sq2):
    d_mu = mu1 - mu2
    d_mu = d_mu[0] * d_mu[0] + d_mu[1] * d_mu[1]
    sum1 = s1 + s2 - 2 * sqrtm(np.linalg.multi_dot([sq1, s2, sq1]))
    # sum2 = s1 + s2 - 2 * sqrtm(np.linalg.multi_dot([sq2, s1, sq2]))
    # tr1 = sum1[0, 0] + sum1[1, 1]
    # tr2 = sum2[0, 0] + sum2[1, 1]
    # print(tr1, tr2, tr1 - tr2)
    return d_mu + sum1[0, 0] + sum1[1, 1]


class InstanceData2(InstanceData):
    def __init__(self, pdata: dict, rdata: dict, tdata: dict, sdata=None, dataset_index=-1):
        super().__init__(pdata, rdata, tdata, sdata)
        self._fix_zstops()
        self._izstops = []
        self._zindex = {}
        self._stop_zid = {}
        self._znames = []
        for k, v in self._fixed_zstops.items():
            zid = len(self._izstops)
            self._zindex[k] = zid
            self._izstops.append(v)
            self._znames.append(k)
            for s in v:
                self._stop_zid[s] = zid
        self._zdistr = None
        self._zdistance = None
        self._dataset_index = dataset_index
        self._zone_locations = None

    def _fix_zstops(self):
        self.zone_stops()
        nan_stops = None
        fixed_zstops = {}
        for k, v in self._zone_stops.items():
            if k != k:
                nan_stops = v
            else:
                fixed_zstops[k] = v
        if nan_stops is None:
            self._fixed_zstops = self._zone_stops
            return
        # set to nearest zone
        if len(nan_stops) == 1:
            s = nan_stops[0]
            n = np.argsort(self._mat[s])
            target_idx = int(n[1])
            if target_idx == s:
                target_idx = int(n[0])
            target = self.stop_zone(target_idx)
            assert target == target
            fixed_zstops[target].append(s)
        else:
            N = self.stop_count()
            for s in nan_stops:
                n = np.argsort(self._mat[s])
                for i in range(1, N):
                    target = self.stop_zone(int(n[i]))
                    if target == target:
                        fixed_zstops[target].append(s)
                        break
        self._fixed_zstops = fixed_zstops

    def dataset_index(self):
        return self._dataset_index

    def zone_count(self):
        return len(self._izstops)

    def zstops(self):
        '''
        list(list(sid))
        zone stops with `nan` fixed
        '''
        return self._izstops

    def stop_zid(self, sid):
        '''zid of a stop'''
        z = self.stop_zone(sid)
        return self._zindex[z]

    def stop_zid_fixed(self, sid):
        '''zid of a stop'''
        return self._stop_zid[sid]

    def zone_id(self, zname):
        return self._zindex[zname]

    def zone_name(self, zid):
        return self._znames[zid]

    def zone_names(self):
        return self._znames

    def common_zones(self, other):
        '''common zone names'''
        result = []
        other_zindex = other._zindex
        for k in self._zindex.keys():
            if other_zindex.get(k) is not None:
                result.append(k)
        return result

    def common_zone_count(self, other):
        return len(self.common_zones(other))

    def _zone_distribution(self):
        zstops = self.zstops()
        locations = self.stop_locations()
        ret = []
        for v in zstops:
            n = len(v)
            if n == 1:
                ret.append([locations[v[0]], ZERO2, ZERO2])
            else:
                loc = locations[v]
                co = np.cov(loc, rowvar=False)
                ret.append([np.mean(loc, axis=0), co, sqrtm(co)])
        return ret

    def zone_distributions(self):
        '''list([mu, sigma, sigma_sqrt])'''
        if self._zdistr is None:
            self._zdistr = self._zone_distribution()
        return self._zdistr

    def zone_distance_mat(self):
        '''distance matrix of zones'''
        if self._zdistance is None:
            distr = self.zone_distributions()
            n = len(distr)
            zdistance = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    vi = distr[i]
                    vj = distr[j]
                    d = normal_distance(vi[0], vi[1], vi[2], vj[0], vj[1], vj[2])
                    zdistance[i, j] = d
                    zdistance[j, i] = d
            self._zdistance = zdistance
        return self._zdistance

    def zone_distance(self, zid, other, other_zid):
        '''distance of two zones'''
        d1 = self.zone_distributions()[zid]
        d2 = other.zone_distributions()[other_zid]
        return normal_distance(d1[0], d1[1], d1[2], d2[0], d2[1], d2[2])

    def zero_distance_stops(self):
        '''stops with zero distance'''
        raw_mat = self.distance_mat()
        mat = raw_mat + np.eye(raw_mat.shape[0])
        mat = mat < 0.01
        where = np.argwhere(mat)
        single_zero = []
        double_zero = []
        for j in range(where.shape[0]):
            i1 = where[j][0]
            i2 = where[j][1]
            d = raw_mat[i2, i1]
            if d > 0.01:
                # i1->i2 = 0, i2->i1 > 0
                single_zero.append([i1, i2])
            else:
                if not [i2, i1] in double_zero:
                    double_zero.append([i1, i2])
        return double_zero, single_zero

    def stop_location_np(self, sid):
        return self.stop_locations()[sid]

    def station_location_np(self):
        return self.stop_locations()[self._station_index]

    def is_station_at(self, loc_np):
        loc1 = self.station_location_np()
        d = loc1 - loc_np
        d = d * d
        return d[0] < 0.01 and d[1] < 0.01

    def is_same_station(self, other):
        return self.is_station_at(other.station_location_np())

    @staticmethod
    def classify_by_station(ins_list):
        station_loc = []
        station_classified = []
        for ins in ins_list:
            find = False
            for i in range(len(station_loc)):
                loc = station_loc[i]
                if ins.is_station_at(loc):
                    station_classified[i].append(ins)
                    find = True
                    break
            if not find:
                station_loc.append(ins.station_location_np())
                station_classified.append([ins])
        return station_classified, station_loc

    def find_rough_similar(self, station_classified, station_loc):
        '''find stops with same station and common zones'''
        same_station = None
        for i in range(len(station_loc)):
            loc = station_loc[i]
            if self.is_station_at(loc):
                same_station = station_classified[i]
                break
        if not same_station:
            return [], []
        result = []
        common_zone_count = []
        for ins in same_station:
            count = self.common_zone_count(ins)
            if count > 0:
                result.append(ins)
                common_zone_count.append(count)
        return result, common_zone_count

    def score(self, path):
        from score import score, route2list
        p = []
        names = self.stop_names()
        for i in range(len(path)):
            sid = path[i]
            p.append(names[sid])
        p.append(p[0])
        return score(route2list({'actual': self.sdata}), p, self.tdata)

    # def plot(self, path_data=None):
    #     RC_VIZ.viz_ins_data(self, path_data)

    # @staticmethod
    # def plot_end(no_show=False):
    #     RC_VIZ.viz_show(no_show)

    def compute_zone_sequence_np(self, path):
        '''compute zone sequence of a path'''
        nz = self.zone_count()
        idx = []
        for i in range(nz):
            idx.append([])
        station = self.station_index()
        for i, s in enumerate(path):
            if s != station:
                idx[self.stop_zid_fixed(s)].append(i)
        for i in range(nz):
            idx[i] = np.mean(idx[i])
        return np.argsort(idx)

    def compute_zone_sequence(self, path):
        '''compute zone sequence of a path'''
        arr = self.compute_zone_sequence_np(path)
        result = []
        for i in range(arr.shape[0]):
            result.append(arr[i])
        return result

    def actual_zone_sequence(self):
        act = self.actual_sequence()
        if act is None:
            return None
        return self.compute_zone_sequence(act)

    def _zone_location(self, zid):
        stops = self.zstops()[zid]
        loc = self.stop_locations()[stops]
        return loc.mean(axis=0)

    def zone_locations(self):
        if self._zone_locations is None:
            loc = []
            for i in range(self.zone_count()):
                loc.append(self._zone_location(i))
            self._zone_locations = loc
        return self._zone_locations

    def simple_solve_zone_sequence(self, weight=None, init=None):
        zloc = self.zone_locations()
        nz = self.zone_count()
        sloc = self.stop_location_np(self.station_index())
        mat = np.zeros((nz + 1, nz + 1))
        for i in range(nz):
            mat[i, i] = np.inf
            d = zloc[i] - sloc
            d = np.sqrt(d[0] * d[0] + d[1] * d[1])
            mat[i, nz] = d
            mat[nz, i] = d
            for j in range(i):
                d = zloc[i] - zloc[j]
                d = np.sqrt(d[0] * d[0] + d[1] * d[1])
                if weight is not None:
                    w = weight[i, j]
                    d = d * np.exp(-w * 1)
                mat[i, j] = d
                mat[j, i] = d
        from tsp import TS
        solver = TS(mat, taboo_size=5, fixed_head=nz, init=init)
        for i in range(150):
            solver.iter()
        path, _ = solver.get_result()
        return path

    def simple_solve_zone_sequence2(self, ref_edges):
        from seq_distance import med_ratio

        zloc = self.zone_locations()
        nz = self.zone_count()
        sloc = self.stop_location_np(self.station_index())

        weight = np.zeros((nz, nz))
        for v in ref_edges:
            ins, edges = v[0], v[1]  # type: InstanceData2
            is_same_station = self.is_same_station(ins)
            common = self.common_zones(ins)
            ncommon = len(common)
            # if is_same_station and ncommon == self.zone_count() and ncommon == ins.zone_count():
            #     zscore = med_ratio(self.actual_zone_sequence(), ins.actual_zone_sequence())
            #     print('find same zones:', self.dataset_index(), ins.dataset_index(), 'zscore:', zscore)
            # return ins.actual_zone_sequence()
            trust = 1.0 if is_same_station else 0.6
            trust *= (ncommon / self.zone_count()) * (ncommon / ins.zone_count())
            for edge in edges:
                weight[edge[0], edge[1]] += trust
        # print('weight max:', np.max(weight))

        mat = np.zeros((nz + 1, nz + 1))
        for i in range(nz):
            mat[i, i] = np.inf
            d = zloc[i] - sloc
            d = np.sqrt(d[0] * d[0] + d[1] * d[1])
            mat[i, nz] = d
            mat[nz, i] = d
            for j in range(i):
                d = zloc[i] - zloc[j]
                d = np.sqrt(d[0] * d[0] + d[1] * d[1])
                w = weight[i, j]
                # d = d * np.exp(-w * 1)
                if w > 0.01:
                    d = d * np.exp(-w / 4. + np.log(0.5))
                mat[i, j] = d
                mat[j, i] = d
        from tsp import TS
        solver = TS(mat, taboo_size=5, fixed_head=nz)
        for i in range(150):
            solver.iter()
        path, _ = solver.get_result()
        return path

    def solve_zone_sequence_train_1(self, data):
        nz = self.zone_count()
        nstop = data['nstop']
        has_window = data['has_window']
        dist_mean = data['dist_mean']
        dist_std = data['dist_std']
        his_same_station_count = data['his_same_station_count']
        his_diff_station_count = data['his_diff_station_count']
        his_same_station_ratio = data['his_same_station_ratio']
        his_diff_station_ratio = data['his_diff_station_ratio']
        ret = []
        for i in range(nz):
            for j in range(i):
                ssc = his_same_station_count[i, j] - 1.0
                dsc = his_diff_station_count[i, j] - 1.0
                if ssc > 0.01 or dsc > 0.01:
                    ssr = his_same_station_ratio[i, j]
                    dsr = his_diff_station_ratio[i, j]
                    nstop1, nstop2 = nstop[i], nstop[j]
                    has_window1, has_window2 = has_window[i], has_window[j]
                    dist_mean1, dist_mean2 = dist_mean[i], dist_mean[j]
                    dist_std1, dist_std2 = dist_std[i], dist_std[j]
                    ret.append([ssc, dsc, ssr, dsr,
                                nstop1, nstop2,
                                has_window1, has_window2,
                                dist_mean1, dist_mean2,
                                dist_std1, dist_std2])
        return ret

    def solve_zone_sequence_train_2(self, data, weights):
        from seq_distance import med_ratio

        nz = self.zone_count()
        his_same_station_count = data['his_same_station_count']
        his_diff_station_count = data['his_diff_station_count']
        # his_same_station_ratio = data['his_same_station_ratio']
        # his_diff_station_ratio = data['his_diff_station_ratio']
        mat = np.copy(data['mat'])
        for i in range(nz + 1):
            mat[i, i] = np.inf
        idx = 0
        for i in range(nz):
            for j in range(i):
                ssc = his_same_station_count[i, j] - 1.0
                dsc = his_diff_station_count[i, j] - 1.0
                d = mat[i, j]
                if ssc > 0.01 or dsc > 0.01:
                    d = d * weights[idx]
                    idx += 1
                mat[i, j] = d
                mat[j, i] = d
        from tsp import TS
        solver = TS(mat, taboo_size=5, fixed_head=nz)
        for i in range(150):
            solver.iter()
        zpath, _ = solver.get_result()
        spath = self.simple_solve(zpath)
        score = self.score(spath)
        zscore = med_ratio(self.actual_zone_sequence(), zpath)
        return score, zscore

    def collect_solve_zseq_inputs(self, ref_edges):
        zloc = self.zone_locations()
        nz = self.zone_count()
        sloc = self.stop_location_np(self.station_index())

        # num stops of each zone
        nstop = np.zeros((nz,))
        # if there is a window in zone
        has_window = np.zeros((nz,))
        dist_mean = np.zeros((nz,))
        dist_std = np.zeros((nz,))
        # need to divide total count
        his_same_station_count = np.zeros((nz, nz))
        his_diff_station_count = np.zeros((nz, nz))
        his_same_station_ratio = np.zeros((nz, nz))
        his_diff_station_ratio = np.zeros((nz, nz))
        for v in ref_edges:
            ins, edges = v[0], v[1]  # type: InstanceData2
            is_same_station = self.is_same_station(ins)
            common = self.common_zones(ins)
            ncommon = len(common)
            for edge in edges:
                a, b = edge[0], edge[1]
                common_ration = (ncommon / self.zone_count()) * (ncommon / ins.zone_count())
                if is_same_station:
                    his_same_station_count[a, b] += 1
                    his_same_station_ratio[a, b] += common_ration
                else:
                    his_diff_station_count[a, b] += 1
                    his_diff_station_ratio[a, b] += common_ration

        for i in range(nz):
            nstop[i] = len(self.zstops()[i])
            if self.is_zone_has_winsow(i):
                has_window[i] = 1
            else:
                has_window[i] = 0

        mat = np.zeros((nz + 1, nz + 1))
        for i in range(nz):
            # mat[i, i] = np.inf
            d = zloc[i] - sloc
            d = np.sqrt(d[0] * d[0] + d[1] * d[1])
            mat[i, nz] = d
            mat[nz, i] = d
            for j in range(i):
                d = zloc[i] - zloc[j]
                d = np.sqrt(d[0] * d[0] + d[1] * d[1])
                mat[i, j] = d
                mat[j, i] = d
        for i in range(nz):
            dists = mat[i, :nz]
            sum = np.sum(dists)
            sum_sq = np.sum(np.square(dists))
            dist_mean[i] = sum / (nz - 1)
            dist_std[i] = np.sqrt(sum_sq / (nz - 1) - dist_mean[i] * dist_mean[i])
        ret = {
            'nstop': nstop,
            'has_window': has_window,
            'dist_mean': dist_mean,
            'dist_std': dist_std,
            'his_same_station_count': his_same_station_count,
            'his_diff_station_count': his_diff_station_count,
            'his_same_station_ratio': his_same_station_ratio,
            'his_diff_station_ratio': his_diff_station_ratio,
            'mat': mat,
        }
        return ret

    def is_zone_has_winsow(self, zid):
        stops = self.zstops()[zid]
        for s in stops:
            packages = self.stop_packages(s)
            for p in packages:
                start, end = p.window()
                if start is not None or end is not None:
                    return True
        return False

    def simple_solve(self, zone_seq=None):
        from tsp import TS, GA
        # from timer import Timer
        # t = Timer()
        result = [self.station_index()]
        if zone_seq is None:
            zone_seq = self.simple_solve_zone_sequence()
        z0 = zone_seq[0]
        zone_div = [[z0]]
        zone_div_count = [len(self.zstops()[z0])]
        target_stop_count = 15
        for i in range(1, len(zone_seq)):
            zid = zone_seq[i]
            stops = self.zstops()[zid]
            nstops = len(stops)
            if zone_div_count[-1] + nstops <= target_stop_count:
                zone_div[-1].append(zid)
                zone_div_count[-1] += nstops
            else:
                zone_div.append([zid])
                zone_div_count.append(nstops)
        last_stop = self.station_index()
        for zones in zone_div:
            ss = []
            for zid in zones:
                ss.append(self.zstops()[zid])
            stops = copy.deepcopy(ss[0])  # type: list
            for j in range(1, len(ss)):
                stops.extend(ss[j])
            stops.append(last_stop)
            sub_mat = self._mat[stops][:, stops]
            solver = TS(sub_mat, taboo_size=5, fixed_head=len(stops) - 1)
            for i in range(50):
                solver.iter()
            path, _ = solver.get_result()
            for i in path:
                result.append(stops[i])
            last_stop = stops[path[-1]]
        # t.tic()
        station = self.station_index()
        stops = [i for i in range(station)]
        for i in range(self.station_index() + 1, self.stop_count()):
            stops.append(i)
        init = []
        for i in range(1, len(result)):
            s = result[i]
            if s > station:
                init.append(s - 1)
            else:
                init.append(s)
        # solver = GA(self._mat, group_size=20, fixed_head=self.station_index(), init=init)
        solver = TS(self._mat, taboo_size=5, fixed_head=self.station_index(), init=init)
        for i in range(100):
            solver.iter()
        path, _ = solver.get_result()
        result = [station]
        for i in path:
            result.append(stops[i])
        # t.tic()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(solver.iter_cost)
        return result

    def propose(self, path):
        proposed = {}
        names = self.stop_names()
        for i, sid in enumerate(path):
            proposed[names[sid]] = i
        return proposed


class DataReader2(DataReader):
    def __init__(self, path, training=False):
        super().__init__(path, training)

    def instance(self, id_or_index):
        if isinstance(id_or_index, int):
            id_or_index = self.package_reader.index_keys[id_or_index]
        pdata = self.package_reader.get(id_or_index)
        rdata = self.route_reader.get(id_or_index)
        tdata = self.travel_reader.get(id_or_index)
        sdata = None
        idx = id_or_index
        id = id_or_index
        if isinstance(idx, int):
            id = self.package_reader.index_keys[id_or_index]
        else:
            idx = self.package_reader.index_keys.index(idx)
        if self.sdata:
            sdata = self.sdata[id]
        return InstanceData2(pdata, rdata, tdata, sdata, dataset_index=idx)


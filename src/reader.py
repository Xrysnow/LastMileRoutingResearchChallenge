import json
from collections import OrderedDict
from typing import BinaryIO
import time
import numpy as np

ROUTE_ID_SIZE = len("RouteID_15baae2d-bf07-4967-956a-173d4036613f")
PACKAGE_ID_SIZE = len("PackageID_07017709-2ddd-4c6a-8b7e-ebde70a4f0fa")


def search_file(file: BinaryIO, pattern, buffer_size=1024 * 1024):
    result = set()
    pattern_size = len(pattern)
    curr = 0
    while True:
        file.seek(curr, 0)
        buf = file.read(buffer_size)
        if len(buf) == 0:
            break
        find_all = []
        find_idx = buf.find(pattern)
        while find_idx != -1:
            find_all.append(find_idx)
            result.add(find_idx + curr)
            find_idx = buf.find(pattern, find_idx + 1)
        curr = curr + buffer_size - pattern_size - 1
    return sorted(list(result))


def search_route(file: BinaryIO, buffer_size=1024 * 1024):
    idx = search_file(file, b'"RouteID_', buffer_size)
    index = OrderedDict()
    size = file.seek(0, 2)
    for i in range(len(idx)):
        start = idx[i] + ROUTE_ID_SIZE + 4
        if i == len(idx) - 1:
            end = size - 1
        else:
            end = idx[i + 1] - 2
        file.seek(idx[i] + 1)
        route_id = file.read(ROUTE_ID_SIZE).decode()
        index[route_id] = [start, end - start]
    return index


def str2sec(x):
    h, m, s = x.strip().split('-')
    return int(h) * 3600 + int(m) * 60 + int(s)


class PackageData:
    def __init__(self, data: dict, id: str):
        self.data = data
        self._id = id
        start_time = data['time_window']['start_time_utc']
        if str(start_time) == 'nan':
            self.start_time = None
        else:
            self.start_time = time.mktime(time.strptime(start_time, '%Y-%m-%d %H:%M:%S'))
        end_time = data['time_window']['end_time_utc']
        if str(end_time) == 'nan':
            self.end_time = None
        else:
            self.end_time = time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M:%S'))
        self._service_time = data['planned_service_time_seconds']
        dim = data['dimensions']
        self._capacity = dim['depth_cm'] * dim['height_cm'] * dim['width_cm']
        # DELIVERED/DELIVERY_ATTEMPTED
        self._status = None
        if data.get('scan_status'):
            self._status = data['scan_status']

    def id(self):
        return self._id

    def window(self):
        return self.start_time, self.end_time

    def service_time(self):
        return self._service_time

    def capacity(self):
        return self._capacity

    def status(self):
        return self._status


class InstanceData:
    def __init__(self, pdata: dict, rdata: dict, tdata: dict, sdata=None):
        self.pdata = pdata
        self.rdata = rdata
        self.tdata = tdata
        self.sdata = sdata['actual'] if sdata else None
        self._stops = sorted([k for k in pdata.keys()])
        self.stop_index = {}
        for i in range(len(self._stops)):
            s = self._stops[i]
            self.stop_index[s] = i
            if self.rdata['stops'][s]['type'] == 'Station':
                self._station = s
                self._station_index = i
        #
        nstops = len(self._stops)
        mat = np.zeros((nstops, nstops))
        for i1, k1 in enumerate(self._stops):
            for i2, k2 in enumerate(self._stops):
                mat[i1, i2] = tdata[k1][k2]
        self._mat = mat
        self._locations = None
        self._zone_stops = None
        #
        self._seq = None
        self._name_seq = None
        if self.sdata:
            seq = []
            name_seq = []
            for name, idx in sorted(self.sdata.items(), key=lambda kv: (kv[1], kv[0])):
                assert len(seq) == idx
                seq.append(self.stop_index[name])
                name_seq.append(name)
            self._seq = seq
            self._name_seq = name_seq
            # always start from station
            assert self._station_index == seq[0]
        #
        self._score_level = None
        if rdata.get('route_score'):
            self._score_level = rdata['route_score']

    def stop_count(self):
        return len(self._stops)

    def stop_names(self):
        '''stop name list'''
        return self._stops

    def departure_time(self):
        return time.mktime(
            time.strptime(self.rdata['date_YYYY_MM_DD'] + ' ' + self.rdata['departure_time_utc'],
                          '%Y-%m-%d %H:%M:%S'))

    def executor_capacity(self):
        return self.rdata['executor_capacity_cm3']

    def _get_stop_data(self, sid):
        if isinstance(sid, int):
            sid = self._stops[sid]
        return self.rdata['stops'][sid]

    def _get_package_data(self, sid):
        if isinstance(sid, int):
            sid = self._stops[sid]
        return self.pdata[sid]

    def stop_location(self, sid):
        '''lat, lng'''
        sdata = self._get_stop_data(sid)
        return sdata['lat'], sdata['lng']

    def stop_locations(self):
        '''array([lat, lng])'''
        if self._locations is None:
            n = self.stop_count()
            locations = np.zeros((n, 2))
            for i in range(n):
                lat, lng = self.stop_location(i)
                locations[i, 0] = lat
                locations[i, 1] = lng
            self._locations = locations
        return self._locations

    def stop_zone(self, sid):
        sdata = self._get_stop_data(sid)
        return sdata['zone_id']

    def zone_stops(self):
        '''dict(zone, list(sid))'''
        if self._zone_stops is None:
            self._zone_stops = {}
            for i in range(self.stop_count()):
                if i == self.station_index():
                    continue
                z = self.stop_zone(i)
                if not self._zone_stops.get(z):
                    self._zone_stops[z] = []
                self._zone_stops[z].append(i)
        return self._zone_stops

    def station_index(self):
        '''departure stop id'''
        return self._station_index

    def station(self):
        return self._stops[self._station_index]

    def stop_packages(self, sid):
        data = self._get_package_data(sid)
        packages = []
        for k, v in data.items():
            packages.append(PackageData(v, k))
        return packages

    def stop_service_time(self, sid):
        '''total service time of a stop'''
        packages = self.stop_packages(sid)
        t = 0.0
        for p in packages:
            t += p.service_time()
        return t

    def distance_mat(self) -> np.ndarray:
        '''time distance matrix'''
        return self._mat.copy()

    def distance(self, s1, s2):
        if isinstance(s1, str):
            s1 = self.stop_index[s1]
        if isinstance(s2, str):
            s2 = self.stop_index[s2]
        return self._mat[s1][s2]

    def actual_sequence(self):
        return self._seq

    def score_level(self):
        return self._score_level


class _BaseDataReader:
    def __init__(self, path):
        self.file = open(path, mode='rb')
        self.file.seek(0)
        #
        self.index = search_route(self.file)
        self.index_keys = [k for k in self.index.keys()]
        self.file.seek(0)
        #
        self._last_key = None
        self._last_get = None

    def get(self, route_id) -> dict:
        '''get content dict'''
        if isinstance(route_id, int):
            route_id = self.index_keys[route_id]
        if self._last_key == route_id:
            return self._last_get
        idx = self.index[route_id]
        self.file.seek(idx[0])
        content = json.loads(self.file.read(idx[1]).decode())
        self.file.seek(0)
        self._last_key = route_id
        self._last_get = content
        return content

    def get_raw(self, route_id):
        if isinstance(route_id, int):
            route_id = self.index_keys[route_id]
        if self._last_key == route_id:
            return self._last_get
        idx = self.index[route_id]
        self.file.seek(idx[0])
        content = self.file.read(idx[1])
        self.file.seek(0)
        return content


class PackageDataReader(_BaseDataReader):
    def __init__(self, path):
        super().__init__(path)


class RouteDataReader(_BaseDataReader):
    def __init__(self, path):
        super().__init__(path)


class TravelDataReader(_BaseDataReader):
    def __init__(self, path):
        super().__init__(path)


class DataReader:
    def __init__(self, path, training=False):
        self.package_reader = PackageDataReader(path + 'package_data.json')
        self.route_reader = RouteDataReader(path + 'route_data.json')
        self.travel_reader = TravelDataReader(path + 'travel_times.json')
        self.sdata = None
        if training:
            with open(path + 'actual_sequences.json') as file:
                self.sdata = json.load(file)
            with open(path + 'invalid_sequence_scores.json') as file:
                self.idata = json.load(file)

    def instance_count(self):
        return len(self.package_reader.index_keys)

    def instance_ids(self):
        return self.package_reader.index_keys

    def instance_id(self, idx):
        return self.package_reader.index_keys[idx]

    def instance(self, id_or_index):
        pdata = self.package_reader.get(id_or_index)
        rdata = self.route_reader.get(id_or_index)
        tdata = self.travel_reader.get(id_or_index)
        sdata = None
        if self.sdata:
            if isinstance(id_or_index, int):
                id_or_index = self.package_reader.index_keys[id_or_index]
            sdata = self.sdata[id_or_index]
        return InstanceData(pdata, rdata, tdata, sdata)


if __name__ == '__main__':
    t = time.perf_counter()
    # reader = DataReader('../data/model_apply_inputs/new_')
    reader = DataReader('../data/model_build_inputs/')
    print(time.perf_counter() - t)
    c = reader.instance(0)
    print(c.departure_time())
    print(c.distance(0, 5))
    print(c.stop_service_time(0))

import random
import math
import numpy as np


class _TSPSolver:
    def __init__(self, mat, fixed_head=None, fixed_tail=None):
        self._mat = mat
        num = mat.shape[0]
        for i in range(num):
            self._mat[i, i] = np.inf
        self.num_city = num
        self._fixed_head = None
        self._fixed_tail = None
        idx_to_remove = []
        if fixed_head is not None:
            self.num_city -= 1
            self._fixed_head = mat[fixed_head, :]
            idx_to_remove.append(fixed_head)
        if fixed_tail is not None:
            self.num_city -= 1
            self._fixed_tail = mat[:, fixed_tail]
            idx_to_remove.append(fixed_tail)
        if len(idx_to_remove):
            self._mat = np.delete(self._mat, idx_to_remove, axis=0)
            self._mat = np.delete(self._mat, idx_to_remove, axis=1)
        #
        self._best_cost = math.inf
        self._best_path = None

    def random_init(self, num_total):
        tmp = [x for x in range(self.num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def compute_cost(self, path):
        cost = 0
        if self._fixed_head is not None:
            cost += self._fixed_head[int(path[0])]
        if self._fixed_tail is not None:
            cost += self._fixed_tail[int(path[-1])]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            cost += self._mat[a][b]
        return cost

    def iter(self):
        raise NotImplementedError()

    def get_result(self):
        return self._best_path, self._best_cost


class ACO(_TSPSolver):
    '''蚁群'''

    def __init__(self, mat, group_size=50, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)
        self.m = group_size  # 蚂蚁数量
        self.alpha = 1  # 信息素重要程度因子
        self.beta = 5  # 启发函数重要因子
        self.rho = 0.1  # 信息素挥发因子
        self.Q = 1  # 常量系数
        num_city = self.num_city
        self.Tau = np.zeros([num_city, num_city])  # 信息素矩阵
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]  # 生成的蚁群
        self.Eta = 10. / self._mat  # 启发式函数
        self.paths = None  # 蚁群中每个个体的长度
        # 存储
        self.iter_cost = []
        # self.greedy_init(self._mat,100,self.num_city)

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        result = result[index]
        for i in range(len(result) - 1):
            s = result[i]
            s2 = result[i + 1]
            self.Tau[s][s2] = 1
        self.Tau[result[-1]][result[0]] = 1
        # for i in range(num_city):
        #     for j in range(num_city):
        # return result

    # 轮盘赌选择
    def rand_choose(self, p):
        x = np.random.rand()
        i = 0
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    # 生成蚁群
    def get_ants(self, num_city):
        for i in range(self.m):
            start = np.random.randint(num_city - 1)
            self.Table[i][0] = start
            unvisit = list([x for x in range(num_city) if x != start])
            current = start
            j = 1
            while len(unvisit) != 0:
                P = []
                # 通过信息素计算城市之间的转移概率
                for v in unvisit:
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)
                P_sum = sum(P)
                P = [x / P_sum for x in P]
                # 轮盘赌选择一个一个城市
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_cost(one)
            result.append(length)
        return result

    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        paths = self.compute_paths(self.Table)
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] = delta_tau[a][b] + self.Q / paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def iter(self):
        # 生成新的蚁群
        self.get_ants(self.num_city)  # out>>self.Table
        self.paths = self.compute_paths(self.Table)
        # 取该蚁群的最优解
        tmp_lenth = min(self.paths)
        tmp_path = self.Table[self.paths.index(tmp_lenth)]
        # 更新最优解
        if tmp_lenth < self._best_cost:
            self._best_cost = tmp_lenth
            self._best_path = tmp_path
        # 更新信息素
        self.update_Tau()
        # 保存结果
        self.iter_cost.append(self._best_cost)
        return tmp_path, tmp_lenth


class DP(_TSPSolver):
    '''动态规划，只进行一次计算'''

    def __init__(self, mat, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)

    def iter(self):
        if self._best_path is not None:
            return self._best_path, self._best_cost
        restnum = [x for x in range(1, self.num_city)]
        tmppath = [0]
        tmplen = 0
        while len(restnum) > 0:
            c = restnum[0]
            restnum = restnum[1:]
            if len(tmppath) <= 1:
                tmppath.append(c)
                tmplen = self.compute_cost(tmppath)
                continue

            insert = 0
            minlen = math.inf
            for i, num in enumerate(tmppath):
                a = tmppath[-1] if i == 0 else tmppath[i - 1]
                b = tmppath[i]
                # tmp1 = self._mat[c][a]
                tmp1 = self._mat[a][c]
                tmp2 = self._mat[c][b]
                curlen = tmplen + tmp1 + tmp2 - self._mat[a][b]
                if curlen < minlen:
                    minlen = curlen
                    insert = i

            tmppath = tmppath[0:insert] + [c] + tmppath[insert:]
            tmplen = minlen
        self._best_path = tmppath
        self._best_cost = tmplen
        return self._best_path, self._best_cost


class GA(_TSPSolver):
    def __init__(self, mat, group_size=30, choose_ratio=0.2, mutate_ratio=0.05,
                 init=None, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)
        self.group_size = group_size
        self.scores = []
        self.ga_choose_ratio = choose_ratio
        self.mutate_ratio = mutate_ratio
        if init == 'random':
            self.fruits = self.random_init(group_size)
        elif init == 'greedy' or init is None:
            self.fruits = self.greedy_init(self._mat, group_size, self.num_city)
        else:
            self.fruits = [init] * group_size
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        # init_best = self.fruits[sort_index[0]]
        # init_best = self.location[init_best]
        # 存储
        self.iter_cost = [1. / scores[sort_index[0]]]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_cost(fruit)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order
        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)
        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)
        assert len(x_conflict_index) == len(y_confict_index)
        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp
        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]
        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        index1 = 0
        index2 = 0
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.group_size:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_cost(gene_x_new)
            y_adp = 1. / self.compute_cost(gene_y_new)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)
        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def iter(self):
        tmp_best_one, tmp_best_score = self.ga()
        tmp_best_cost = 1. / tmp_best_score
        self.iter_cost.append(tmp_best_cost)
        if tmp_best_cost < self._best_cost:
            self._best_cost = tmp_best_cost
            self._best_path = tmp_best_one
        return tmp_best_one, tmp_best_cost


class PSO(_TSPSolver):
    def __init__(self, mat, group_size=200, init=None, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)
        self.num = group_size  # 粒子数目
        # 初始化所有粒子
        # if random_init:
        #     self.particals = self.random_init(self.num)
        # else:
        #     self.particals = self.greedy_init(self._mat, num_total=self.num, num_city=self.num_city)
        if init == 'random':
            self.particals = self.random_init(self.num)
        elif init == 'greedy' or init is None:
            self.particals = self.greedy_init(self._mat, num_total=self.num, num_city=self.num_city)
        else:
            self.fire = [init] * group_size
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        # init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self._best_cost = self.global_best_len
        self._best_path = self.global_best
        # 存储每次迭代的结果
        self.iter_cost = [init_l]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_cost(one)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_cost(one)
        one2 = cross_part + tmp
        l2 = self.compute_cost(one2)
        if l1 < l2:
            return one, l1
        else:
            return one, l2

    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_cost(one)
        return one, l2

    def iter(self):
        # 更新粒子群
        for i, one in enumerate(self.particals):
            tmp_l = self.lenths[i]
            # 与当前个体局部最优解进行交叉
            new_one, new_l = self.cross(one, self.local_best[i])
            if new_l < self._best_cost:
                self._best_cost = tmp_l
                self._best_path = one

            if new_l < tmp_l or np.random.rand() < 0.1:
                one = new_one
                tmp_l = new_l

            # 与当前全局最优解进行交叉
            new_one, new_l = self.cross(one, self.global_best)

            if new_l < self._best_cost:
                self._best_cost = tmp_l
                self._best_path = one

            if new_l < tmp_l or np.random.rand() < 0.1:
                one = new_one
                tmp_l = new_l
            # 变异
            one, tmp_l = self.mutate(one)

            if new_l < self._best_cost:
                self._best_cost = tmp_l
                self._best_path = one

            if new_l < tmp_l or np.random.rand() < 0.1:
                one = new_one
                tmp_l = new_l

            # 更新该粒子
            self.particals[i] = one
            self.lenths[i] = tmp_l
        # 评估粒子群，更新个体局部最优和个体当前全局最优
        self.eval_particals()
        # 更新输出解
        if self.global_best_len < self._best_cost:
            self._best_cost = self.global_best_len
            self._best_path = self.global_best
        self.iter_cost.append(self._best_cost)
        return self.global_best, self.global_best_len


class SA(_TSPSolver):
    '''模拟退火'''

    def __init__(self, mat, T0=4000, Tend=1e-3, rate=0.9995, init=None, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)
        self.T0 = T0
        self.Tend = Tend
        self.rate = rate
        if init == 'random':
            self.fire = self.random_init(1)[0]
        elif init == 'greedy' or init is None:
            self.fire = self.greedy_init(self._mat, 100, self.num_city)
        else:
            self.fire = init
        self._best_path = self.fire
        self._best_cost = self.compute_cost(self.fire)
        # 存储
        self.iter_cost = [self._best_cost]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        return result[index]

    # 计算一个温度下产生的一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_cost(one)
            result.append(length)
        return result

    # 产生一个新的解：随机交换两个元素的位置
    def get_new_fire(self, fire):
        fire = fire.copy()
        t = [x for x in range(len(fire))]
        a, b = np.random.choice(t, 2)
        fire[a:b] = fire[a:b][::-1]
        return fire

    # 退火策略，根据温度变化有一定概率接受差的解
    def eval_fire(self, raw, get, temp):
        len1 = self.compute_cost(raw)
        len2 = self.compute_cost(get)
        dc = len2 - len1
        p = max(1e-1, np.exp(-dc / temp))
        if len2 < len1:
            return get, len2
        elif np.random.rand() <= p:
            return get, len2
        else:
            return raw, len1

    def iter(self):
        if self.T0 <= self.Tend:
            return self._best_cost, self._best_path
        # 产生在这个温度下的随机解
        tmp_new = self.get_new_fire(self.fire.copy())
        # 根据温度判断是否选择这个解
        self.fire, fire_cost = self.eval_fire(self._best_path, tmp_new, self.T0)
        # 更新最优解
        if fire_cost < self._best_cost:
            self._best_cost = fire_cost
            self._best_path = self.fire
        # 降低温度
        self.T0 *= self.rate
        # 记录
        self.iter_cost.append(self._best_cost)
        return self.fire, fire_cost


class TS(_TSPSolver):
    '''禁忌搜索'''

    def __init__(self, mat, taboo_size=5, init=None, fixed_head=None, fixed_tail=None):
        super().__init__(mat, fixed_head, fixed_tail)
        self.taboo_size = taboo_size
        self.taboo = []
        if init == 'random':
            self.path = self.random_init(1)[0]
        elif init == 'greedy' or init is None:
            self.path = self.greedy_init(self._mat, 100, self.num_city)
        else:
            self.path = init
        self.cur_path = self.path
        self._best_path = self.path
        self._best_cost = self.compute_cost(self.path)
        # 存储
        self.iter_cost = [self._best_cost]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                if tmp_choose == -1:
                    tmp_choose = rest[0]
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        return result[index]
        # return result[0]

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_cost(one)
            result.append(length)
        return result

    # 产生随机解
    def ts_search(self, x):
        moves = []
        new_paths = []
        while len(new_paths) < 400:
            i = np.random.randint(len(x))
            j = np.random.randint(len(x))
            tmp = x.copy()
            tmp[i:j] = tmp[i:j][::-1]
            new_paths.append(tmp)
            moves.append([i, j])
        return new_paths, moves

    def iter(self):
        new_paths, moves = self.ts_search(self.cur_path)
        new_lengths = self.compute_paths(new_paths)
        sort_index = np.argsort(new_lengths)
        min_l = new_lengths[sort_index[0]]
        min_path = new_paths[sort_index[0]]
        min_move = moves[sort_index[0]]
        # 更新当前的最优路径
        if min_l < self._best_cost:
            self._best_cost = min_l
            self._best_path = min_path
            self.cur_path = min_path
            # 更新禁忌表
            if min_move in self.taboo:
                self.taboo.remove(min_move)
            self.taboo.append(min_move)
        else:
            # 找到不在禁忌表中的操作
            while min_move in self.taboo and len(sort_index) > 1:
                sort_index = sort_index[1:]
                min_path = new_paths[sort_index[0]]
                min_move = moves[sort_index[0]]
            self.cur_path = min_path
            self.taboo.append(min_move)
        # 禁忌表超长了
        if len(self.taboo) > self.taboo_size:
            self.taboo = self.taboo[1:]
        self.iter_cost.append(self._best_cost)
        return min_path, min_l

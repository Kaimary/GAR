import os.path
import warnings
from typing import Dict, List

import click
import numpy as np
# Using the dot product in the sparse module can speed up set intersection operation drastically
from scipy import sparse
from time import time

# Some magic numbes
_stepsize = 0.1
_largenumber = 1E5
_smallnumber = 1E-5


class SetCover:
    """
    Set Cover Problem - Find a set of columns that cover all the rows with minimal cost.

    Algorithm:
        -- greedy: 
        -- Lagrangian relaxation:
    Input: 
        -- a_matrix[mrows, ncols], the covering binary matrix, 
           a_matrix[irow, jcol] = True if jcol covers irow
        -- cost[ncols], the cost of columns. 
           I recommend using normalized cost: cost/median(cost)

    (Use A.4 instance from Beasley's OR library,
     http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html, as an example)
    Instantiation: 
        >> a_matrix = np.load('./BeasleyOR/scpa4_matrix.npy')
        >> cost = np.load('./BeasleyOR/scpa4_cost.npy')
        >> g = setcover.SetCover(a_matrix, cost)
    Run the solver: 
        >> solution, time_used = g.SolveSCP()

    Output:
        -- g.s, the (near-optimal) minimal set of columns, a binary 1D array, 
           g.s[jcol] = True if jcol is in the solution
        -- g.total_cost, the total cost of the (near-optimal) minimal set of columns

    Comments:
        -- 

    References:
        -- Guangtun Ben Zhu, 2016
           A New View of Classification in Astronomy with the Archetype Technique: 
           An Astronomical Case of the NP-complete Set Cover Problem
           AJ/PASP, (to be submitted)
        -- Caprara, A., Fischetti, M. and Toth, P., 1999 
           A heuristic method for the set covering problem 
           Operations research, 47(5), pp.730-743.
        -- Fisher, M.L., 2004 
           The Lagrangian relaxation method for solving integer programming problems 
           Management science, 50(12_supplement), pp.1861-1871.
        -- Balas, E. and Carrera, M.C., 1996 
           A dynamic subgradient-based branch-and-bound procedure for set covering 
           Operations Research, 44(6), pp.875-890.
        -- Beasley, J.E., 1990  
           OR-Library: distributing test problems by electronic mail  
           Journal of the operational research society, pp.1069-1072.
        -- And many more, please see references in Zhu (2016).

    To_do:
        -- Better converging criteria needed (to cut time for small instances)
    History:
        -- 14-May-2016, Alternate initialization methods between random and greedy, BGT, JHU
        -- 19-Apr-2016, Documented, BGT, JHU
        -- 10-Dec-2015, Started, BGT, JHU
        -- DD-MMM-2010, Conceived, BGT, NYU
    """

    def __init__(self, amatrix, cost, maxiters=20, subg_nsteps=15, subg_maxiters=100):
        """
        Initialization
        Required argument:
            amatrix
            cost
        Keywords:
            maxiters - the maximum number of iterations for the re-initialization, 
                       20 by default. This is the parameter you may want to increase
                       to get a better solution (at the expense of time)
            subg_nsteps - how many steps for each subgradient iteration, 15 by default
                          Note each step includes _subg_nadaptive (hard-coded at 20) mini steps
            subg_maxiters - the maximum number of iterations for the subgradient phase, 
                            100 by default

        """
        self.a = np.copy(amatrix)
        # Compressed sparse row
        self.a_csr = sparse.csr_matrix(amatrix, copy=True)
        # Compressed sparse column (transposed for convenience)
        self.a_csc = sparse.csr_matrix(amatrix.T, copy=True)
        self.c = np.copy(cost)
        self.mrows = amatrix.shape[0]
        self.ncols = amatrix.shape[1]

        # Some Magic Numbers based on the Beasley's test bed

        ## subgradient method magic numbers
        self._stepsize = _stepsize
        # how many steps we look back to decide whether to increase or decrease the stepsize
        self._subg_nadaptive = 20
        self._subg_nsteps = self._subg_nadaptive * subg_nsteps
        # Maximum iterations we want to perturb the best u and then recalculate
        self._subg_maxiters = subg_maxiters
        self._subg_maxfracchange = 0.000020  # convergence criteria, fractional change
        self._subg_maxabschange = 0.010  # convergence criteria, absolute change
        self._max_adapt = 0.06  # threshold to half the stepsize
        self._min_adapt = 0.002  # threshold to increase the stepsize by 1.5
        self._u_perturb = 0.06  # perturbations

        ## re-initialization magic numbers
        self._maxiters = maxiters
        self._maxfracchange = 0.001  # convergence criterion, fractional change
        self._LB_maxfracchange = 0.050  # update criterion for Lower Bound

        # setting up
        self.f_uniq = self._fix_uniq_col()  # fix unique columns
        self.f = np.copy(self.f_uniq)  # fixed columns, only using f_uniq for now
        self.f_covered = np.any(self.a[:, self.f], axis=1)  # rows covered by fixed columns
        self.s = np.copy(self.f_uniq)  # (current) solution, selected column
        self.u = self._u_naught()  # (current best) Lagrangian multiplier

    @property
    def total_cost(self):
        """
        Total cost of a given set s
        """
        return np.einsum('i->', self.c[self.s])

    @property
    def fixed_cost(self):
        """
        Total cost of a given set s
        """
        return np.einsum('i->', self.c[self.f])

    def reset_all(self):
        """
        Reset the parameters to start over
        """
        self._stepsize = _stepsize
        self.reset_f()
        self.reset_s()
        self.reset_u()

    def reset_s(self):
        """
        Reset s, the selected columns
        """
        self.s = np.copy(self.f_uniq)  # (current) solution, selected column

    def reset_f(self):
        """
        Reset f, the fixed columns
        """
        self.f = np.copy(self.f_uniq)
        self.f_covered = np.any(self.a[:, self.f], axis=1)

    def reset_u(self, random=False):
        """
        Reset u, the Lagrangian multiplier
        """
        if (random):
            self.u = self._u_naught_simple()
        else:
            self.u = self._u_naught()

    def _u_naught_simple(self):
        """
        Initial guess of the Lagrangian multiplier with random numbers
        """
        # Random is better to give different multipliers in the subgradient phase
        return np.random.rand(self.mrows) * 1.

    def _u_naught(self):
        """
        Initial guess of the Lagrangian multiplier with greedy algorithm
        This is the default initializer
        """
        adjusted_cost = self.c / self.a_csc.dot(np.ones(self.mrows))
        cost_matrix = adjusted_cost * self.a + np.amax(adjusted_cost) * (~self.a)
        return adjusted_cost[np.argmin(cost_matrix, axis=1)]

    def _fix_uniq_col(self):
        """
        Fix the unique columns that have to be in the minimal set
        """
        # subgradient; for two boolean arrays, multiplication seems to be the best way 
        # (equivalent to logical_and)
        n_covered_col = self.a_csr.dot(np.ones(self.ncols))
        ifix = np.zeros(self.ncols, dtype=bool)
        if (np.count_nonzero(n_covered_col) != self.mrows):
            raise ValueError("There are uncovered rows! Please check your input!")
        if (np.any(n_covered_col == 1)):
            inonzero = self.a_csr[n_covered_col == 1, :].nonzero()
            ifix[inonzero[1]] = True

        return ifix

    def greedy(self, u=None, niters_max=1000):
        """
        Heuristic greedy method to select a set of columns to cover all the rows
        start from the initial set
        run the following first if you want to reset the initial selection with fixed columns
            - self.reset_s() or 
            - self.s = np.logical_or(self.s, self.f)
        """

        niters = 1
        if (u is None):
            u = self.u

        utmp = np.copy(u)
        iuncovered = ~np.any(self.a[:, self.s], axis=1)

        score = np.zeros(self.ncols)
        while (np.count_nonzero(iuncovered) > 0) and (niters <= niters_max):
            # It's 5 times faster without indexing, the advantage is made possible by csc_matrix.dot
            mu = (self.a_csc.dot((iuncovered).astype(int))).astype(float)
            mu[mu <= _smallnumber] = _smallnumber

            utmp[~iuncovered] = 0
            gamma = (self.c - self.a_csc.dot(utmp))
            select_gamma = (gamma >= 0)

            if (np.count_nonzero(select_gamma) > 0):
                score[select_gamma] = gamma[select_gamma] / mu[select_gamma]
            if (np.count_nonzero(~select_gamma) > 0):
                score[~select_gamma] = gamma[~select_gamma] * mu[~select_gamma]

            inewcolumn = (np.nonzero(~self.s)[0])[np.argmin(score[~self.s])]
            self.s[inewcolumn] = True
            iuncovered = ~np.logical_or(~iuncovered, self.a[:, inewcolumn])
            niters = niters + 1
        if (niters == niters_max):
            warnings.warn("Iteration in Greedy reaches maximum = {0}".format(niters_max))
        return self.total_cost

    def update_core(self):
        """
        Removing fixed columns
        """
        if (~np.any(self.f)):
            a_csr = sparse.csr_matrix(self.a, copy=True)  # Compressed sparse row
            a_csc = sparse.csr_matrix(self.a.T, copy=True)  # Compressed sparse column (transposed)
        else:
            a_csr = sparse.csr_matrix(self.a[:, ~self.f][~self.f_covered, :], copy=True)
            a_csc = sparse.csr_matrix(self.a[:, ~self.f][~self.f_covered, :].T, copy=True)
        return (a_csr, a_csc)

    def subgradient(self):
        """
        Subgradient step for the core problem N\S. 
        """

        UB_full = self.total_cost
        ufull = np.copy(self.u)

        # Update core: possible bottleneck
        (a_csr, a_csc) = self.update_core()
        mrows = a_csr.shape[0]
        ncols = a_csr.shape[1]
        u_this = self.u[~self.f_covered]
        # np.einsum is 20% faster than np.sum ...
        UB_fixed = self.fixed_cost
        UB = UB_full - UB_fixed
        cost = self.c[~self.f]

        # save nsteps calculations (Lagrangian multipliers and lower bounds)
        u_sequence = np.zeros((mrows, self._subg_nsteps))
        Lu_sequence = np.zeros(self._subg_nsteps)
        # update u
        x = np.zeros(ncols, dtype=bool)
        niters_max = self._subg_maxiters
        maxfracchange = self._subg_maxfracchange
        maxabschange = self._subg_maxabschange

        # initialization
        f_change = _largenumber
        a_change = _largenumber
        niters = 0
        Lu_max0 = 0
        while ((f_change > maxfracchange) or (a_change > maxabschange)) and (niters < niters_max):
            u_this = (1.0 + (np.random.rand(mrows) * 2. - 1) * self._u_perturb) * u_this
            u_sequence[:, 0] = u_this
            cost_u = cost - a_csc.dot(u_sequence[:, 0])  # Lagrangian cost
            # next lower bound of the Lagrangian subproblem
            Lu_sequence[0] = np.einsum('i->', cost_u[cost_u < 0]) + np.einsum('i->', u_sequence[:, 0])

            for i in np.arange(self._subg_nsteps - 1):
                # current solution to the Lagrangian subproblem
                x[0:] = False
                x[cost_u < 0] = True

                # subgradient; for two boolean arrays, multiplication seems to be the best way 
                # (equivalent to logical_and)
                s_u = 1. - a_csr.dot(x.astype(int))
                s_u_norm = np.einsum('i,i', s_u, s_u)  # subgradient's norm squared

                # Update
                # next Lagrangian multiplier
                u_temp = u_sequence[:, i] + self._stepsize * (UB - Lu_sequence[i]) / s_u_norm * s_u
                u_temp[u_temp < 0] = 0

                u_sequence[:, i + 1] = u_temp
                cost_u = cost - a_csc.dot(u_sequence[:, i + 1])  # Lagrangian cost
                # next lower bound of the Lagrangian subproblem
                Lu_sequence[i + 1] = np.einsum('i->', cost_u[cost_u < 0]) + np.einsum('i->', u_sequence[:, i + 1])

                # print(UB_full, UB, Lu_sequence[i+1])
                # Check the last nadaptive steps and see if the step size needs to be adapted
                if (np.mod(i + 1, self._subg_nadaptive) == 0):
                    Lu_max_adapt = np.amax(Lu_sequence[i + 1 - self._subg_nadaptive:i + 1])
                    Lu_min_adapt = np.amin(Lu_sequence[i + 1 - self._subg_nadaptive:i + 1])
                    if (Lu_max_adapt <= 0.):
                        Lu_max_adapt = _smallnumber
                    f_change_adapt = (Lu_max_adapt - Lu_min_adapt) / np.fabs(Lu_max_adapt)
                    if f_change_adapt > self._max_adapt:
                        self._stepsize = self._stepsize * 0.5
                    elif (f_change_adapt < self._min_adapt) and (self._stepsize < 1.5):
                        self._stepsize = self._stepsize * 1.5
                    # swap the last multiplier with the optimal one
                    i_optimal = np.argmax(Lu_sequence[i + 1 - self._subg_nadaptive:i + 1])
                    if (i_optimal != (self._subg_nadaptive - 1)):
                        u_temp = u_sequence[:, i]
                        u_sequence[:, i] = u_sequence[:, i + 1 - self._subg_nadaptive + i_optimal]
                        u_sequence[:, i + 1 - self._subg_nadaptive + i_optimal] = u_temp
                        Lu_sequence[i + 1 - self._subg_nadaptive + i_optimal] = Lu_sequence[i]
                        Lu_sequence[i] = Lu_max_adapt

            i_optimal = np.argmax(Lu_sequence)
            Lu_max = Lu_sequence[i_optimal]
            u_this = u_sequence[:, i_optimal]
            niters = niters + 1
            a_change = Lu_max - Lu_max0
            f_change = a_change / np.fabs(Lu_max)
            Lu_max0 = Lu_max  # Just a copy. Not the reference (It's a number object)
            # save current u_this???

            if (niters == niters_max):
                warnings.warn("Iteration in subgradient reaches maximum = {0}".format(niters))

        # update multipliers
        self.u[~self.f_covered] = u_this

        # return the last nsteps multipliers
        # save nsteps calculations (Lagrangian multipliers and lower bounds)
        u_sequence_full = np.zeros((self.mrows, self._subg_nsteps))
        Lu_sequence_full = np.zeros(self._subg_nsteps)
        u_sequence_full[self.f_covered, :] = self.u[self.f_covered][:, np.newaxis]
        u_sequence_full[~self.f_covered, :] = u_sequence

        Lu_sequence_full = Lu_sequence + self.fixed_cost

        return (u_sequence_full, Lu_sequence_full)

    def subgradient_solution(self, u=None):
        """
        """
        if (u is None):
            u = np.copy(self.u)
        cost_u = self.c - self.a_csc.dot(u)  # Lagrangian cost
        x = np.zeros(self.ncols, dtype=bool)  # has to be integer to use scipy sparse matrix
        x[cost_u < 0] = True  # current solution to the Lagrangian subproblem
        return x

    def SolveSCP(self):
        """
        The wrapper, Solve the SCP
        """

        t0 = time()

        # Some predicates
        Lu_min = 0.
        niters_max = self._maxiters
        maxfracchange = self._maxfracchange

        # initialization, resetting ...
        self.reset_all()  # including _u_naught(), first application
        scp_min = self.greedy()

        # re-initialization iteration; col fixing ignored for the moment
        niters = 0
        f_change = _largenumber
        while (f_change > maxfracchange) and (niters < niters_max):
            # re-initialize u
            if (np.mod(niters, 2) == 0):
                self.reset_u(random=True)
            else:
                self.reset_u()
            u_tmp, Lu_tmp = self.subgradient()  # find a near-optimal solution
            u, Lu = self.subgradient()  # rerun subgradient to get a set of Lagrangian multipliers

            scp_all = np.zeros(self._subg_nsteps)
            for i in np.arange(self._subg_nsteps):
                # self.reset_s()
                self.s = np.copy(self.f)
                scp_all[i] = self.greedy(u=u[:, i])

            # check if the solution is gettting better
            imin_tmp = (np.where(scp_all == np.amin(scp_all)))[0]
            imin = imin_tmp[np.argmax(Lu[imin_tmp])]
            imax = np.argmax(Lu)
            if (np.mod(niters, 5) == 0):
                print("This Best solution: UB={0}, LB={1}, UB1={2}, LB1={3}".format(scp_all[imin], Lu[imin],
                                                                                    scp_all[imax], Lu[imax]))
            if (niters == 0) or (
                    (scp_all[imin] <= scp_min) and ((Lu[imin] - Lu_min) > -(np.fabs(Lu_min) * self._LB_maxfracchange))):
                scp_min = scp_all[imin]
                u_min = np.copy(u[:, imin])
                Lu_min = Lu[imin]
                self.stepsize = _stepsize

            LB = Lu_min

            # final step, needs to get u_min back
            self.u = np.copy(u_min)
            self.s = np.copy(self.f)
            UB = self.greedy()

            # Which is better? absolute change or fractional change? 
            # Both are fine, but cost should be normalized over the mean/median.
            GAP = (UB - LB) / np.fabs(UB)
            f_change = GAP
            if (np.mod(niters, 5) == 0):
                print("Current Best Solution: UB={0}, LB={1}, change={2}% @ niters={3}".format(UB, LB, f_change * 100.,
                                                                                               niters))
            niters = niters + 1
            if (niters == niters_max):
                # warnings.warn("Iteration reaches maximum = {0}".format(niters))
                print("Iteration in re-initialization reaches maximum number = {0}".format(niters))

        # Need to remove redundant columns
        # self.remove_redundant() # this itself is NP-hard ...

        print("Current Best Solution: UB={0}, LB={1}, change={2}% @ niters={3}".format(UB, LB, f_change * 100., niters))
        print("Final Best solution: {0}".format(UB))
        time_used = (time() - t0) / 60.
        print("Took {0:.3f} minutes to reach current solution.".format(time_used))

        return (UB, time_used)


def generate_dev_gold(dev_json: List[Dict], dev_gold_path: str):
    """
    generate separated dev gold sql file
    @param dev_json: separated dev json list
    @param dev_gold_path: dev gold sql save path
    @return: None
    """
    dev_gold_sql_list = list()
    # Iter every dev item
    for dev_item in dev_json:
        dev_gold_sql_list.append(dev_item['query'] + '\t' + dev_item['db_id'] + '\n')
    # save dev gold sql
    with open(dev_gold_path, 'w') as f:
        for dev_gold_sql in dev_gold_sql_list:
            f.write(dev_gold_sql)
    return


dev_set_name = ['world_1', 'employee_hire_evaluation', 'museum_visit', 'battle_death', 'voter_1',
                'cre_Doc_Template_Mgt', 'orchestra', 'network_1', 'concert_singer', 'car_1', 'course_teach',
                'poker_player', 'student_transcripts_tracking', 'dog_kennels', 'wta_1', 'pets_1', 'tvshow', 'singer',
                'flight_2', 'real_estate_properties']


@click.command()
@click.argument("input_file_name", type=click.Path(exists=True, dir_okay=False))
@click.argument("dev_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", type=click.Path(exists=False, dir_okay=True))
def set_cover(input_file_name: str, dev_path: str, out_dir: str):
    import json
    import random

    """
    uni_set format:
    elem_i_j: i-th DB's j-th unique element
    
    [[elem_0_0, elem_0_1, ..., elem_0_n0],
    [elem_1_0, elem_1_1, ..., elem_1_n1],
    [elem_2_0, elem_2_1, ..., elem_2_n2],
    ..................................,
    [elem_k_0, elem_k_1, ..., elem_k_nk]]
    """
    uni_set = list()

    """
    subset_set format:
    elem_i_j_k: i-th DB's j-th set's k-th element

    [[[elem_0_0_0, ..., elem_0_0_k0],
    [elem_0_1_0, ..., elem_0_1_k1],
    ..............................,
    [elem_0_n0_0, ..., elem_0_n0_kn0]],
    [[elem_1_0_0, ..., elem_1_0_k0],
    [elem_1_1_0, ..., elem_1_1_k1],
    ..............................,
    [elem_1_n0_0, ..., elem_1_n0_kn0]],
    ...................................,
    [[elem_l_0_0, ..., elem_l_0_k0],
    [elem_l_1_0, ..., elem_l_1_k1],
    ..............................,
    [elem_l_n0_0, ..., elem_l_n0_kn0]]]
    """
    subset_set = list()

    """
    key_set format:
    sql_i_j: i-th DB's j-th set's name

    [[sql_0_0, sql_0_1, ..., sql_0_n0],
    [sql_1_0, sql_1_1, ..., sql_1_n1],
    [sql_2_0, sql_2_1, ..., sql_2_n2],
    ..................................,
    [sql_k_0, sql_k_1, ..., sql_k_nk]]
    """
    key_set = list()

    db_name = list()

    with open(input_file_name, 'r') as f:
        data = json.load(f)

        for db, value in data.items():
            if db not in dev_set_name:
                continue
            cur_uniset = set()
            cur_subsetSet = list()
            cur_keyset = list()

            db_name.append(db)

            # current DB
            for query, set_ in value.items():
                cur_keyset.append(query)
                cur_subsetSet.append(set_)
                for i, elem in enumerate(set_):
                    cur_uniset.add(elem)

            key_set.append(cur_keyset)
            subset_set.append(cur_subsetSet)
            uni_set.append(cur_uniset)

    db_size = len(db_name)
    uni_set_map = []

    for _, set_ in enumerate(uni_set):
        # UniSetMap[key] = i
        temp = {}
        for i, elem in enumerate(set_):
            temp[elem] = i
        uni_set_map.append(temp)

    amatrixs = []
    costs = []
    for i in range(db_size):
        cur_matrix = np.zeros(shape=(len(uni_set[i]), len(subset_set[i])), dtype=np.bool)
        for col, set_ in enumerate(subset_set[i]):
            for _, elem in enumerate(set_):
                row = uni_set_map[i][elem]
                cur_matrix[row][col] = True
        amatrixs.append(cur_matrix)
        costs.append(np.ones(len(subset_set[i])))

    result_set = []
    # [result number, set number]
    cnt_per_db = []

    for i in range(db_size):
        solve = SetCover(amatrixs[i], costs[i])
        solve.SolveSCP()

        cur_result = []
        for j, key in enumerate(key_set[i]):
            if solve.s[j]:
                cur_result.append(key)
        result_set.append(cur_result)
        cnt_per_db.append([len(cur_result), len(key_set[i])])

    ret = {}
    result = {}
    # with open("log.txt", 'w') as f:
    print("[INFO]")
    for i, db in enumerate(db_name):
        ret[db] = [int(sql[sql.find('@') + 1:]) for sql in result_set[i]]
        result[db] = cnt_per_db[i][0] / cnt_per_db[i][1]
        print(f"{db}:\n\tresult set size:{cnt_per_db[i][0]}\n\ttotal set size:"
              f"{cnt_per_db[i][1]}\n\t{cnt_per_db[i][0]/cnt_per_db[i][1]}\n")

    dev_dict = {}
    with open(dev_path, 'r') as f:
        data = json.load(f)
        for ex in data:
            db_id = ex["db_id"]
            if db_id not in dev_dict.keys():
                dev_dict[db_id] = list()
            dev_dict[db_id].append(ex)

    dev_1 = list()
    dev_2 = list()
    for db in db_name:
        if db in dev_dict.keys():
            if result[db] > 0.8:
                ret[db] = random.sample(ret[db], int(len(dev_dict[db]) * 0.8))
            l = [dev_dict[db][numb] for numb in ret[db]]
            dev_1.extend(l)
            for ex in dev_dict[db]:
                if ex not in l:
                    dev_2.append(ex)

    dev_1_path = os.path.join(out_dir, 'dev_80.json')
    dev_1_gold_path = os.path.join(out_dir, 'dev_gold_80.sql')
    with open(dev_1_path, 'w') as f1:
        json.dump(dev_1, f1, indent=4)
    generate_dev_gold(dev_1, dev_1_gold_path)

    dev_2_path = os.path.join(out_dir, 'dev_20.json')
    dev_2_gold_path = os.path.join(out_dir, 'dev_gold_20.sql')
    with open(dev_2_path, 'w') as f2:
        json.dump(dev_2, f2, indent=4)
    generate_dev_gold(dev_2, dev_2_gold_path)


if __name__ == '__main__':
    set_cover()

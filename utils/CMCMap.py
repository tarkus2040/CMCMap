import os

import numpy as np

import graphviz

from utils.causal_simplex import CMC_simplex, DCMC_simplex

class CMCMap:

    def __init__(self, df, tau=2, emd=8, bivCMC_thres=0.5, dcmc_thres=0.5, **kwargs):
        
        self.df = df # the dataframe, extract the column names or indices, determine the causal graph
        self.kwargs = kwargs # dictionary of other parameters, including the CMC and DCMC parameters
        
        self.tau = tau # time delay for time delay embedding
        self.emd = emd

        if not 0 <= bivCMC_thres <= 1:
            raise ValueError('bivCMC_thres must be between 0 and 1')
        if not 0 <= dcmc_thres <= 1:
            raise ValueError('dcmc_thres must be between 0 and 1')
        self.bivCMC_thres = bivCMC_thres
        self.dcmc_thres = dcmc_thres # threshold for DCMC edge removal

        self.n = df.shape[1] # number of variables
        self.var_names = df.columns # variable names
        self.var_indices = np.arange(self.n) # pool of variable indices

        self.adj_matrix = None # adjacency matrix of the causal graph

        # to save the score stats of the two phases for information
        self.phase1_stats = {}
        self.phase2_stats = {}

    
    def fit(self):
        """ Fit the multivarCM model.
        Returns:
        ch: dict
            the children of each variable.
"""
        ch=self._initial_causal_graph()
        ch=self._eliminate_edges(ch)
        self.ch=ch


        return ch
    
    def get_adj_matrix(self):
        """ Get the adjacency matrix of the causal graph. (to be called after fitting)
        Returns:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph."""
        return self.adj_matrix

    def draw_graph(self, save_path=None):
        """ Draw the causal graph.
        Args:
        ch: dict
            the children of each variable."""
        ch=self.ch
        dot = graphviz.Digraph()
        for k in ch:
            for c in ch[k]:
                cause_name = self.var_names[k]
                effect_name = self.var_names[c]
                dot.edge(cause_name, effect_name)

        # view and save the graph
        if save_path is not None:
            dot.render(os.path.join(save_path, 'causal_graph'), format='png', view=True)

        return dot

    def _initial_causal_graph(self):
        # exhaustive bivariate search for initial causal graph (doesn't distinguish between direct and indirect)
        S=self.var_indices.copy() # pool of variable indices, start with all variables

        # initialize adjacency matrix
        self.adj_matrix=np.zeros((self.n,self.n))

        ch={i: [] for i in range(self.n)} # dictionary to store the children of each variable
        total_pairs = self.n * (self.n - 1) // 2
        pair_index = 0
        print(f'[CMC] Evaluating {total_pairs} variable pairs.', flush=True)

        for i in range(self.n): # cause

            # do not test redundant pairs
            for j in S[S>i]: # effect
                pair_index += 1
                # cross map between the current variable i and the candidate child j
                # determine whether the edge between i and j is redundant
                cause_ind=[i]
                effect_ind=[j]
                cause_list=self.var_names[cause_ind].tolist()
                effect_list=self.var_names[effect_ind].tolist()
                print(
                    f'[CMC] {pair_index}/{total_pairs}: '
                    f'{cause_list[0]} -> {effect_list[0]} / '
                    f'{effect_list[0]} -> {cause_list[0]}',
                    flush=True,
                )

                # key in dictionary to store the stats ("cause -> effect | conds")
                phase1_stats_key=f'causes_{cause_ind} -> effect_{effect_ind}'

                # create the CMC_simplex object
                cm=CMC_simplex(self.df,cause_list,effect_list,self.tau,self.emd,**self.kwargs)
                output=cm.causality() # CMC scores: cause -> effect, effect -> cause


                del cm

                # store the stats
                self.phase1_stats[phase1_stats_key] = output

                score_c2e, score_e2c = output
                if max(score_c2e, score_e2c) < self.bivCMC_thres:
                    continue
                if score_c2e > score_e2c:
                    ch[i].append(j)
                    self.adj_matrix[i,j]=1
                elif score_e2c > score_c2e:
                    ch[j].append(i)
                    self.adj_matrix[j,i]=1

        print('[CMC] Pairwise evaluation complete.', flush=True)
        return ch     
                
        
    def _eliminate_edges(self, ch):
        """ The second step: eliminate redundant edges.
        * Note my definition of ordering is from sink to top.
        
        For each pairwise edge that has other variables in between, this might be indirecto causality"""

        # MultiDCMC
        list_to_remove=[] # will be used to store tuples of edges (cause, effect) to remove
        total_edges = sum(len(children) for children in ch.values())
        edge_index = 0
        print(f'[multiDCMC] Checking {total_edges} candidate edges.', flush=True)

        for i in range(self.n):
            for j in ch[i]: # children of i
                edge_index += 1
                print(
                    f'[multiDCMC] {edge_index}/{total_edges}: '
                    f'{self.var_names[i]} -> {self.var_names[j]}',
                    flush=True,
                )
                # check if a causal path (from adjacency matrix) can be established between i and j
                # if there is a path, do PCM to determine if it is a indirect causation
                    # if it is, remove the edge between i and j
                    # if it is not, keep the edge between i and j
                # if there is no path, keep the edge between i and j
                
                # bool, list_var_on_path = has_path(self.adj_matrix, i, j)
                bool, list_var_on_path = find_longest_path(self.adj_matrix, i, j)

                if bool:
                    # create the DCMC object
                    # cause_list=self.var_names[[i]].tolist()
                    # effect_list=self.var_names[[j]].tolist()
                    # conds_list=[self.var_names[k] for k in list_var_on_path]

                    cause_ind=[i]
                    effect_ind=[j]
                    # conditions: list_var_on_path - i - j
                    conds_ind=[k for k in list_var_on_path if k!=i and k!=j]

                    # skip if there are no conditions
                    if len(conds_ind)==0:
                        continue

                    cause_list=self.var_names[cause_ind].tolist()
                    effect_list=self.var_names[effect_ind].tolist()
                    conds_list=self.var_names[conds_ind].tolist() 

                    # key in dictionary to store the stats ("cause -> effect | conds")
                    phase2_stats_key=f'causes_{cause_ind} -> effect_{effect_ind} | conds_{conds_ind}'

                    dcmc=DCMC_simplex(self.df,cause_list,effect_list,conds_list,self.tau,self.emd,**self.kwargs)
                    output=dcmc.causality() # direct scores: cause -> effect, effect -> cause
                    del dcmc

                    # store the stats
                    self.phase2_stats[phase2_stats_key] = output


                    if output[0] < self.dcmc_thres:
                        list_to_remove.append((i,j))
                else:
                    continue

        # remove the edges
        for edge in list_to_remove:
            i,j=edge
            ch[i].remove(j)
            self.adj_matrix[i,j]=0

        print(
            f'[multiDCMC] Edge checks complete; removed {len(list_to_remove)} edges.',
            flush=True,
        )
        return ch

def find_longest_path(adj_matrix, i, j):
    """Find the longest path of length >= 3 between two variables i and j.

    Args:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph.
        i: int
            the index of the cause variable.
        j: int
            the index of the effect variable.
    
    Returns:
        bool: True if there is a path of length >= 3 between i and j, False otherwise.
        list: The longest path of nodes if a valid path exists, otherwise None.
    """
    n = adj_matrix.shape[0]
    stack = [(i, [i])]  # Stack stores tuples of (current node, path)
    longest_path = []  # Track the longest valid path
    
    while stack:
        node, path = stack.pop()

        if node == j and len(path) >= 3:  # Check if the current path is valid
            if len(path) > len(longest_path):  # Update if it's the longest valid path
                longest_path = path
        
        for k in range(n):
            if adj_matrix[node, k] == 1 and k not in path:  # Avoid revisiting nodes in the same path
                stack.append((k, path + [k]))  # Append the new path

    if longest_path:
        return True, longest_path  # Return the longest valid path
    else:
        return False, None  # No valid path found

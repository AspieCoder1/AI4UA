import itertools
from typing import Generator, Self

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.isomorphism import DiGraphMatcher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Lattice:

    def __init__(self, loe=torch.zeros(1).to(device)):
        self.loe = torch.from_numpy(loe).to(device)
        self.loe_transposed = torch.transpose(self.loe, 0, 1)
        self.adj = None
        self.size = self.loe.size(dim=1)
        self.majority_tensor = None
        self.minority_tensor = None
        self.join_tensor = None
        self.meet_tensor = None
        self.is_a_lattice = False
        self.dist = False
        self.mod = False
        self.meet_semi_dist = False
        self.join_semi_dist = False
        self.semi_dist = False
        # our super cool code contribution 
        # self.agru_n2 = False
        self.quasi_cancel = False

        #compute the matrices of majorities and minorities, for al n,m maj[n,m] = [0,..,0,1,0...] 1 for elements that are >= n,m 0 otherwise
        self.majority_tensor, self.minority_tensor = self.compute_majmin_tensors()
        #compute matrices of join and meet where join_matrix[n,m] = join(n,m)
        self.join_tensor, self.meet_tensor, self.is_a_lattice = self.compute_joinmeet()

        if self.is_a_lattice:
            self.dist = self.is_distributive()
            self.mod = self.is_modular()
            self.meet_semi_dist = self.is_meet_semidistributive()
            self.join_semi_dist = self.is_join_semidistributive()
            self.semi_dist = self.is_semidistributive()
            # self.agru_n2 = self.is_agruesian_n_2()
            self.quasi_cancel = self.is_cancellative()
            self.adj = self.loe2adj()

    def is_distributive(self):
        ### old slowest
        # for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
        #     if self.meet_tensor[x, self.join_tensor[y, z]] != self.join_tensor[self.meet_tensor[x, y], self.meet_tensor[x, z]]:
                # return False
        # return True
        tensor_size = [self.size,self.size,self.size]
        ### left-side: x & (y | z)
        # x:
        x = torch.tensor(np.arange(self.size)).to(device)
        x_tensor = x.repeat_interleave(self.size*self.size, dim=0).reshape(tensor_size)
        # y | z:
        y_join_z_tensor = self.join_tensor.expand(tensor_size)
        # x & (y | z)
        left_side = self.meet_tensor[x_tensor,y_join_z_tensor]

        ### right-side: (x & y) | (x & z)
        # x & y:
        x_meet_y_tensor = self.meet_tensor.repeat_interleave(self.size, dim=1).reshape(tensor_size)
        # x & z:
        x_meet_z_tensor = x_meet_y_tensor.transpose(1,2)
        # (x & y) | (x & z)
        right_side = self.join_tensor[x_meet_y_tensor, x_meet_z_tensor]
        if not torch.equal(left_side,right_side):
            return False
        return True

    def is_modular(self):
        ### old slowest
        # for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
        #     if self.join_tensor[self.meet_tensor[x, y], self.meet_tensor[z, y]] != self.meet_tensor[self.join_tensor[self.meet_tensor[x, y], z], y]:
                # return False
        # return True

        tensor_size = [self.size, self.size, self.size]
        ### left-side: (x & y) | (x & z)
        # x & y:
        x_meet_y_tensor = self.meet_tensor.repeat_interleave(self.size, dim=1).reshape(tensor_size)
        # x & z:
        x_meet_z_tensor = x_meet_y_tensor.transpose(1, 2)
        # (x & y) | (x & z)
        left_side = self.join_tensor[x_meet_y_tensor, x_meet_z_tensor]

        ### right-side: ((x & y) | z) & x
        # x & y:
        # z:
        z = torch.tensor(np.arange(self.size)).to(device)
        z_tensor = z.repeat(self.size, 1).expand(tensor_size)
        # (x & y) | z:
        x_meet_y__join_z_tensor = self.join_tensor[x_meet_y_tensor,z_tensor]
        # x:
        x_tensor = z.repeat_interleave(self.size*self.size, dim=0).reshape(tensor_size)
        # x & ((x & y) | z):
        right_side = self.meet_tensor[x_tensor, x_meet_y__join_z_tensor]

        if not torch.equal(left_side,right_side):
            return False
        return True

    def is_meet_semidistributive(self):
        # old slow
        # for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
        #     if self.meet_tensor[x, y] == self.meet_tensor[x, z] and self.meet_tensor[x, self.join_tensor[y, z]] != self.meet_tensor[x, y]:
        #         return False
        # return True

        tensor_size = [self.size, self.size, self.size]

        # condition 1: (x & y) == (x & z)
        # x & y:
        x_meet_y_tensor = self.meet_tensor.repeat_interleave(self.size, dim=1).reshape(tensor_size)
        # x & z:
        x_meet_z_tensor = x_meet_y_tensor.transpose(1, 2)
        condition_1 = x_meet_y_tensor == x_meet_z_tensor

        # condition 2: (x & y) == (x & (y | z))
        # x:
        x = torch.tensor(np.arange(self.size)).to(device)
        x_tensor = x.repeat_interleave(self.size * self.size, dim=0).reshape(tensor_size)
        # y | z:
        y_join_z_tensor = self.join_tensor.expand(tensor_size)
        # x & (y | z)
        x_meet__y_join_z_tensor = self.meet_tensor[x_tensor, y_join_z_tensor]
        condition_2 = x_meet_y_tensor == x_meet__y_join_z_tensor

        if torch.any(torch.logical_and(condition_1, torch.logical_not(condition_2))):
            return False
        return True

    def is_join_semidistributive(self):
        # for (x, y, z) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
        #     if self.join_tensor[x, y] == self.join_tensor[x, z] and self.join_tensor[x, self.meet_tensor[y, z]] != self.join_tensor[x, y]:
        #         return False
        # return True

        tensor_size = [self.size, self.size, self.size]

        # condition 1: (x | y) == (x | z)
        # x | y:
        x_join_y_tensor = self.join_tensor.repeat_interleave(self.size, dim=1).reshape(tensor_size)
        # x | z:
        x_join_z_tensor = x_join_y_tensor.transpose(1, 2)
        condition_1 = x_join_y_tensor == x_join_z_tensor

        # condition 2: (x | y) == (x | (y & z))
        # x:
        x = torch.tensor(np.arange(self.size)).to(device)
        x_tensor = x.repeat_interleave(self.size * self.size, dim=0).reshape(tensor_size)
        # y & z:
        y_meet_z_tensor = self.meet_tensor.expand(tensor_size)
        # x | (y & z)
        x_join__y_meet_z_tensor = self.join_tensor[x_tensor, y_meet_z_tensor]
        condition_2 = x_join_y_tensor == x_join__y_meet_z_tensor

        if torch.any(torch.logical_and(condition_1, torch.logical_not(condition_2))):
            return False
        return True

    def is_semidistributive(self):
        if self.is_join_semidistributive() and self.is_meet_semidistributive():
            return True
        return False
    
    def is_agruesian_n_2(self):
        ### old slowest
        for (a, b, c, d) in itertools.product(range(1, self.size-1), range(1, self.size-1), range(1, self.size-1), range(1, self.size-1)):
            LHS = self.meet_tensor[a, self.join_tensor[b, self.join_tensor[c, d]]]
            RHS = self.join_tensor[c, self.meet_tensor[self.join_tensor[c, a], self.join_tensor[b, d]]]
            if self.meet_tensor[LHS, RHS] != LHS:
                return False
        return True
        # tensor_size = [self.size,self.size,self.size,self.size]
        # a = torch.tensor(np.arange(self.size)).to(device)
        # a_tensor = a.repeat_interleave(self.size ** 3 , dim=0).reshape(tensor_size)
        # b = torch.tensor(np.arange(self.size)).to(device)
        # b_tensor = b.repeat_interleave(self.size ** 3 , dim=0).reshape(tensor_size)
        # # c | d:
        # c_join_d_tensor = self.join_tensor.expand(tensor_size)
        # # b | (c | d)
        # b_join_c_join_d = self.join_tensor[b_tensor, c_join_d_tensor]
        # left_side = self.meet_tensor[a_tensor,b_join_c_join_d]

        # c = torch.tensor(np.arange(self.size)).to(device)
        # c_tensor = c.repeat_interleave(self.size ** 3 , dim=0).reshape(tensor_size)
        # a_join_c_tensor = self.join_tensor.expand(tensor_size)
        # b_join_d_tensor = self.join_tensor.expand(tensor_size)
        # ajc_meet_bjd = self.meet_tensor[a_join_c_tensor, b_join_d_tensor]
        # right_side = self.join_tensor[c_tensor, ajc_meet_bjd]

        # lhs_meet_rhs = self.meet_tensor[left_side, right_side]
        # lhs_join_rhs = self.join_tensor[left_side, right_side]
        # if not (torch.equal(left_side,right_side)):
        #     return False
        # return True

    def is_cancellative(self):
        x = torch.tensor(np.arange(self.size)).to(device)
        x = x.repeat_interleave(self.size ** 2 , dim=0)
        
        y = torch.tensor(np.arange(self.size)).to(device)
        y = y.repeat_interleave(self.size ** 1 , dim=0)
        y = torch.cat([y] * self.size ** 1)
        
        z = torch.tensor(np.arange(self.size)).to(device)
        z = torch.cat([z] * self.size ** 2)

        x_meet_z = self.meet_tensor[x, z]
        y_meet_z = self.meet_tensor[y, z]
        meet_eq = torch.eq(x_meet_z, y_meet_z)

        x_join_z = self.join_tensor[x, z]
        y_join_z = self.join_tensor[y, z]
        join_eq = torch.eq(x_join_z, y_join_z)

        both_eq = torch.logical_and(meet_eq, join_eq)

        impl = torch.logical_or(torch.eq(x, y), torch.logical_not(both_eq))
        
        if torch.all(impl):
            return True
        else:
            return False

    def compute_majmin_tensors(self):
        '''
            Compute the majority/minority matrix multipling raws/columns of loe
        '''

        idx_kron = [i*(self.size+1) for i in range(self.size)]

        majority_tensor = torch.kron(self.loe, self.loe)[:, [idx_kron]].reshape(self.size, self.size, self.size)
        minority_tensor = torch.kron(self.loe_transposed, self.loe_transposed)[:, [idx_kron]].reshape(self.size, self.size, self.size)

        return majority_tensor, minority_tensor

    def compute_joinmeet(self):
        '''
            Compute the join/meet matrix multipling loe to majority/minority element rows x columuns and then component-wise
            still to majority/minority element. With these 2 operation a vector which counts the # of majorities grater
            than a majority n is computed and then the join of a pair (a,b) is detached as the majority that as # of majorities
            grater that it = to the corresponding component in the obtained vector.
        '''


        try:
            join_tensor_full = torch.matmul(self.majority_tensor, self.loe)
            join_tensor = (join_tensor_full == 1).nonzero(as_tuple=True)[-1].reshape(self.size, self.size)

            meet_tensor_full = torch.matmul(self.minority_tensor, self.loe_transposed)
            meet_tensor = (meet_tensor_full == 1).nonzero(as_tuple=True)[-1].reshape(self.size, self.size)

            is_a_lattice = True

        except:
            join_tensor = None
            meet_tensor = None
            is_a_lattice = False

        return join_tensor, meet_tensor, is_a_lattice

    def loe2adj(self, reflexive=False):
        # adj = np.copy(self.loe)
        adj = self.loe.detach().clone()
        if not reflexive:
            for i in range(self.size):
                adj[i, i] = 0

        for i in range(self.size):
            for j in range(i + 1, self.size):
                if adj[i, j] == 1:
                    for k in range(j + 1, self.size):
                        if adj[j, k] == 1:
                            adj[i, k] = 0
        return adj

    def __contains__(self, item: Self) -> bool:
        g1 = nx.from_numpy_array(self.adj, create_using=nx.DiGraph)
        g2 = nx.from_numpy_array(item.adj, create_using=nx.DiGraph)
        matcher = DiGraphMatcher(g1, g2)

        return matcher.subgraph_is_isomorphic()

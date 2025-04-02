#!/usr/bin/env python
# coding: utf-8


from scipy.special import binom
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
from scipy import linalg

np.set_printoptions(linewidth=120)


class BasicFermions:
    '''
    Store fermionic slater determinant in the form of a 1/0 vector

    Attributes:
        nsites(int): number of sites (or single particle wavefunctions)
        nparticles(int): number of particles in this determinant
        nconserved(bool): tells if the number of particle is conserved
            True: Hilbert space with 'nparticles' particles / False: Fock space
        array(int): array of 1 or 0 that represents the determinant
        list(int): list of 1 or 0 that represents the determinant
        nspace(int): number of sub-Hilbert space (always 1 for this class)
        max(int): size of the space considered
        index(int): return the unique determinant index in the considered space
        parttype(str): name of the particle (important for commutation rules)
        comrules(dict): name of the particle associated with the phase of the
        commutation
    '''

    def __init__(self, array, nconserved=True, parttype='fermions',
                 comrules={'fermions': -1}):
        '''Constructor of the BasicFermions classe

            Parameters:
            array(int): initial configuration of the determinant
            nconserved(bool): if True(default) the number of particles is
            conserved
            parttype(str): name of the particle (important for commutation
            rules)
            comrules(dict): name of the particle associated with the phase
            of the commutation
        '''
        self.__nconserved = nconserved
        self.__nspace = 1
        self.__parttype = parttype
        self.__comrules = comrules
        self.array = np.array(array)
        self.array.flags.writeable = False

    @classmethod
    def firstdet(cls, nparticles, nsites, nconserved=True, parttype='fermions',
                 comrules={'fermions': -1}):
        if (nconserved):
            array = np.zeros(nsites, dtype=int)
            array[:nparticles] = 1
            return cls(array, nconserved, parttype)
        else:
            array = np.zeros(nsites, dtype=int)
            return cls(array, nconserved, parttype)

    @property
    def nparticles(self):
        return self.__nparticles

    @property
    def nsites(self):
        return self.__nsites

    @property
    def nconserved(self):
        return self.__nconserved

    @property
    def array(self):
        return self.__array

    @property
    def list(self):
        return self.__array.tolist()

    @property
    def nsiteslist(self):
        return [self.__nsites]

    @property
    def nspace(self):
        return self.__nspace

    @property
    def max(self):
        return self.__max

    @property
    def index(self):
        return self.__index

    @property
    def parttype(self):
        return self.__parttype

    @property
    def comrules(self):
        return self.__comrules

    @parttype.setter
    def parttype(self, parttype):
        self.__parttype = parttype

    @array.setter
    def array(self, array):
        if (type(array) == np.ndarray and
                np.array_equal(array, array.astype(bool))):
            self.__array = array
            self.__nparticles = sum(array)
            self.__nsites = self.__array.size
            self.__max = self.__calc_max()
            self.__index = self.__calc_index()
        else:
            raise ValueError("only numpy arrays of integers 0/1 are accepted")

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return (str(self.__class__.__name__) + "(" + str(self.list) +
                ", " + str(self.nconserved) + ")")

    def __eq__(self, other):
        if (type(self) == type(other)):
            return (np.array_equal(self.array, other.array)
                    and self.nconserved == other.nconserved
                    and self.parttype == other.parttype
                    and self.comrules == other.comrules)
        else:
            return False

    def __mul__(self, other):
        return TensorProductState(self, other)

    def copy(self):
        return self.__class__(self.array, self.nconserved)

    def deepcopy(self):
        return self.__class__(self.array, self.nconserved)

    @staticmethod
    def same_space(elem1, elem2):
        if (elem1.nconserved and elem1.nconserved):
            return (type(elem1) == type(elem2)
                    and elem1.nparticles == elem2.nparticles
                    and elem1.nsites == elem2.nsites
                    and elem1.parttype == elem2.parttype
                    and elem1.comrules == elem2.comrules)
        if (not (elem1.nconserved or elem1.nconserved)):
            return (type(elem1) == type(elem2) and elem1.nsites == elem2.nsites
                    and elem1.parttype == elem2.parttype
                    and elem1.comrules == elem2.comrules)
        else:
            return False

    def __calc_index(self):
        if (self.nconserved):
            j = 0
            num = 0
            previ = -1
            for i in range(self.nsites):
                if self.array[i] == 1:
                    j = j + 1
                    for m in range(previ+1, i):
                        num = num + binom(self.nsites-m-1, self.nparticles-j)
                    previ = i
        else:
            num = 0
            for bit in reversed(self.array):
                num = (num << 1) | bit
        return int(num)

    def __calc_max(self):
        if (self.nconserved):
            return int(binom(self.nsites, self.nparticles))
        else:
            return 2**self.nsites

    def reset(self):
        if (self.nconserved):
            tarray = np.zeros(self.nsites, dtype=int)
            tarray[:self.nparticles] = 1
            self.array = tarray.copy()
        else:
            self.array = np.zeros(self.nsites, dtype=int)
            self.__nparticles = 0
        self.__calc_index()
        return self

    def vacuum(self):
        self.array = np.zeros(self.nsites, dtype=int)
        self.__nparticles = 0
        self.__calc_index()
        return self

    def next_element(self):
        tarray = self.array.copy()
        if (self.nconserved):
            nextfree = False
            nbr = 0
            for i in range(self.nsites-1, -1, -1):
                if self.array[i] == 1:
                    nbr = nbr + 1
                    if nextfree:
                        tarray[i:] = 0
                        tarray[i+1:i+1+nbr] = 1
                        self.array = tarray.copy()
                        return True
                else:
                    nextfree = True
            return False
        else:
            for i in range(self.nsites):
                if tarray[i] == 0:
                    tarray[i] = 1
                    tarray[:i] = 0
                    self.array = tarray.copy()
                    return True
            return False

    def range(self, elem1=None, elem2=None):
        if (elem2 is None):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                if (self.__class__.same_space(self, elem1)):
                    elem = self.deepcopy()
                    elem.reset()
                    if (elem.index < elem1.index):
                        while(not np.array_equal(elem.array, elem1.array)):
                            yield elem
                            elem.next_element()
                        return
                else:
                    raise ValueError("range can only end with"
                                     "elements of the same space")
        else:
            if (self.__class__.same_space(elem1, elem2)):
                if (elem1.index < elem2.index):
                    elem = elem1.deepcopy()
                    while(not np.array_equal(elem.array, elem2.array)):
                        yield elem
                        elem.next_element()
                    return
            else:
                raise ValueError("range can only be used between"
                                 "elements of the same space")

    def go_through(self, parttype):
        if (parttype in self.__comrules):
            coeff = (self.__comrules[parttype])**self.__nparticles
            return (self.copy(), coeff)
        else:
            coeff = 1
            return (self.copy(), coeff)

    def operator(self, op):
        tarray = self.array.copy()
        coeff = 1
        ind = op[1]
        if (ind < 0 or ind > self.nsites):
            raise ValueError("position " + str(ind)
                             + " does not exist in this state")
        if op[2] == -1:
            if tarray[ind] == 0:
                return (None, 0)
            elif tarray[ind] == 1:
                coeff = coeff*(-1)**(sum(tarray[ind+1:]))
                tarray[ind] = 0
        elif op[2] == 1:
            if tarray[ind] == 0:
                coeff = coeff*(-1)**(sum(tarray[ind+1:]))
                tarray[ind] = 1
            elif tarray[ind] == 1:
                return (None, 0)
        else:
            return (None, 0)
        return (self.__class__(tarray, self.nconserved), coeff)



class BasicBosons:
    '''
    Store bosonic slater determinant in the form of a 1/0 vector

    Attributes:
        nsites(int): number of sites (or single particle wavefunctions)
        nparticles(int): number of particles in this determinant
        nconserved(bool): tells if the number of particle is conserved
            True: Hilbert space with 'nparticles' particles / False: Fock space
        array(int): array of 1 or 0 that represents the determinant
        list(int): list of 1 or 0 that represents the determinant
        nspace(int): number of sub-Hilbert space (always 1 for this class)
        max(int): size of the space considered
        index(int): return the unique determinant index in the considered space
        parttype(str): name of the particle (important for commutation rules)
        comrules(dict): name of the particle associated with the phase of the
        commutation
    '''

    def __init__(self, array, nmax, ntotlimited=False, ntotmax=0,
                 parttype='bosons', comrules={'bosons': 1}):
        '''Constructor of the BasicFermions classe

            Parameters:
            array(int): initial configuration of the determinant
            nconserved(bool): if True(default) the number of particles is
            conserved
            parttype(str): name of the particle (important for commutation
            rules)
            comrules(dict): name of the particle associated with the phase
            of the commutation
        '''
        if (ntotlimited):
            self.__nmax = ntotmax*np.ones_like(array)
        else:
            self.__nmax = np.array(nmax)
        self.__nspace = 1
        self.__ntotlimited = ntotlimited
        self.__ntotmax = ntotmax
        self.__parttype = parttype
        self.__comrules = comrules
        self.array = np.array(array)
        self.array.flags.writeable = False

    @classmethod
    def firstdet(cls, nsites, nmax, ntotlimited=False, ntotmax=0,
                 parttype='bosons', comrules={'bosons': 1}):
        array = np.zeros(nsites, dtype=int)
        return cls(array, nmax, ntotlimited, ntotmax,
                 parttype, comrules)

    @property
    def nparticles(self):
        return np.sum(self.__array)

    @property
    def nsites(self):
        return self.__nsites

    @property
    def nmax(self):
        return self.__nmax

    @property
    def ntotmax(self):
        return self.__ntotmax

    @property
    def ntotlimited(self):
        return self.__ntotlimited

    @property
    def array(self):
        return self.__array

    @property
    def list(self):
        return self.__array.tolist()

    @property
    def nsiteslist(self):
        return [self.__nsites]

    @property
    def nspace(self):
        return self.__nspace

    @property
    def max(self):
        return self.__max

    @property
    def index(self):
        return self.__index

    @property
    def parttype(self):
        return self.__parttype

    @property
    def comrules(self):
        return self.__comrules

    @parttype.setter
    def parttype(self, parttype):
        self.__parttype = parttype

    @array.setter
    def array(self, array):
        if (self.ntotlimited):
            if (type(array) == np.ndarray and
                    np.sum(array) <= self.ntotmax):
                self.__array = array
                self.__nsites = self.__array.size
                self.__max = self.__calc_max()
                self.__index = self.__calc_index()
            else:
                raise ValueError("only numpy arrays of integers with sum<ntotmax are accepted")
        else:
            if (type(array) == np.ndarray and
                    np.prod(np.less_equal(array, self.nmax))):
                self.__array = array
                self.__nsites = self.__array.size
                self.__max = self.__calc_max()
                self.__index = self.__calc_index()
            else:
                raise ValueError("only numpy arrays of integers <nmax are accepted")

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return (str(self.__class__.__name__) + "(" + str(self.list) +
                ", " + np.array2string(self.nmax, separator=', ') +
                ", " + str(self.ntotlimited) + ", " + str(self.ntotmax) +
                ")")

    def __eq__(self, other):
        if (type(self) == type(other)):
            if (self.ntotlimited):
                return (np.array_equal(self.array, other.array)
                        and self.nmax == other.nmax
                        and self.ntotlimited == other.ntotlimited
                        and self.ntotmax == other.ntotmax
                        and self.parttype == other.parttype
                        and self.comrules == other.comrules)
            else:
                return (np.array_equal(self.array, other.array)
                        and self.nmax == other.nmax
                        and self.ntotlimited == other.ntotlimited
                        and self.parttype == other.parttype
                        and self.comrules == other.comrules)
        else:
            return False

    def __mul__(self, other):
        return TensorProductState(self, other)

    def copy(self):
        return self.__class__(self.array, self.nmax,
                              self.ntotlimited, self.ntotmax)

    def deepcopy(self):
        return self.__class__(self.array, self.nmax,
                              self.ntotlimited, self.ntotmax)

    @staticmethod
    def same_space(elem1, elem2):
        if (elem1.ntotlimited):
            return (type(elem1) == type(elem2)
                    and np.array_equal(elem1.nmax, elem2.nmax)
                    and elem1.nsites == elem2.nsites
                    and elem1.ntotlimited == elem2.ntotlimited
                    and elem1.ntotmax == elem2.ntotmax
                    and elem1.parttype == elem2.parttype
                    and elem1.comrules == elem2.comrules)
        else:
            return (type(elem1) == type(elem2)
                    and np.array_equal(elem1.nmax, elem2.nmax)
                    and elem1.nsites == elem2.nsites
                    and elem1.ntotlimited == elem2.ntotlimited
                    and elem1.parttype == elem2.parttype
                    and elem1.comrules == elem2.comrules)

    def __calc_index(self):
        if (self.ntotlimited):
            nr = self.nparticles
            ns = self.array.size
            num = 0
            for i in range(ns):
                if ((nr-1) >= 0):
                    num += binom(nr-1+ns-i, nr-1)
                    nr -= self.array[i]
                else:
                    break
        else:
            num = self.array[-1]
            for i in range(self.array.size-1):
                num += self.array[i]*np.prod(self.nmax[i+1:]+1)
        return int(num)

    def __calc_max(self):
        if (self.ntotlimited):
            return int(binom(self.ntotmax+self.nsites, self.ntotmax))
        else:
            return np.prod(self.nmax+1)

    def reset(self):
        self.array = np.zeros(self.nsites, dtype=int)
        self.__calc_index()
        return self

    def vacuum(self):
        self.array = np.zeros(self.nsites, dtype=int)
        self.__calc_index()
        return self

    def next_element(self):
        if (self.ntotlimited):
            np = self.nparticles
            tarray = self.array.copy()
            carry = tarray[-1]
            tarray[-1] = 0
            for i in reversed(range(self.nsites-1)):
                if (tarray[i] > 0):
                    tarray[i] = tarray[i] - 1
                    tarray[i+1] = carry + 1
                    self.array = tarray.copy()
                    return True
            if (np < self.ntotmax):
                tarray[:] = 0
                tarray[0] = np + 1
                self.array = tarray.copy()
                return True
            return False
        else:
            tarray = self.array.copy()
            for i in reversed(range(self.nsites)):
                if tarray[i] < self.nmax[i]:
                    tmp=tarray[i] + 1
                    tarray[i:] = 0
                    tarray[i] = tmp
                    self.array = tarray.copy()
                    return True
            return False

    def range(self, elem1=None, elem2=None):
        if (elem2 is None):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                if (self.__class__.same_space(self, elem1)):
                    elem = self.deepcopy()
                    elem.reset()
                    if (elem.index < elem1.index):
                        while(not np.array_equal(elem.array, elem1.array)):
                            yield elem
                            elem.next_element()
                        return
                else:
                    raise ValueError("range can only end with"
                                     "elements of the same space")
        else:
            if (self.__class__.same_space(elem1, elem2)):
                if (elem1.index < elem2.index):
                    elem = elem1.deepcopy()
                    while(not np.array_equal(elem.array, elem2.array)):
                        yield elem
                        elem.next_element()
                    return
            else:
                raise ValueError("range can only be used between"
                                 "elements of the same space")

    def go_through(self, parttype):
        coeff = 1
        return (self.copy(), coeff)

    def operator(self, op):
        tarray = self.array.copy()
        coeff = 1
        ind = op[1]
        if (ind < 0 or ind > self.nsites):
            raise ValueError("position " + str(ind)
                             + " does not exist in this state")
        if op[2] == -1:
            if tarray[ind] == 0:
                return (None, 0)
            else :
                coeff = coeff*np.sqrt(tarray[ind])
                tarray[ind] = tarray[ind]-1
        elif op[2] == 1:
            if (self.ntotlimited):
                if (self.nparticles < self.ntotmax):
                    coeff = coeff*np.sqrt(tarray[ind]+1)
                    tarray[ind] = tarray[ind]+1
                else:
                    return (None, 0)
            else:
                if (tarray[ind] < self.nmax[ind]):
                    coeff = coeff*np.sqrt(tarray[ind]+1)
                    tarray[ind] = tarray[ind]+1
                else:
                    return (None, 0)
        else:
            return (None, 0)
        return (self.__class__(tarray, self.nmax,
                               self.ntotlimited, self.ntotmax), coeff)



class TensorProductState:

    def __init__(self, state1, state2):
        if state2.nspace == 1:
            self.__state1 = state1.deepcopy()
            self.__state2 = state2.deepcopy()
        else:
            self.__state1 = (state1*state2.state1)
            self.__state2 = state2.state2
        self.__nspace = state1.nspace + state2.nspace
        self.__nparticles = state1.nparticles + state2.nparticles
        self.__nsites = state1.nsites + state2.nsites
        self.__index = self.__calc_index()
        self.__max = self.__calc_max()
        if (self.__state1.nspace == 1 and self.__state2.nspace == 1):
            self.__parttype = [self.state1.parttype, self.state2.parttype]
        elif (self.__state1.nspace > 1 and self.__state2.nspace == 1):
            self.__parttype = self.state1.parttype
            self.__parttype.append(self.state2.parttype)
        else:
            self.__parttype = self.state1.parttype + self.state2.parttype

    @property
    def state1(self):
        return self.__state1.copy()

    @property
    def state2(self):
        return self.__state2.copy()

    @property
    def list(self):
        if (self.__state1.nspace == 1 and self.__state2.nspace == 1):
            return [self.__state1.list, self.__state2.list]
        elif (self.__state1.nspace > 1 and self.__state2.nspace == 1):
            return self.__state1.list + [self.__state2.list]
        else:
            self.__state1.list + self.__state2.list

    @property
    def nsiteslist(self):
        if (self.__state1.nspace == 1 and self.__state2.nspace == 1):
            return [self.__state1.nsites, self.__state2.nsites]
        elif (self.__state1.nspace > 1 and self.__state2.nspace == 1):
            return self.__state1.nsiteslist + [self.__state2.nsites]
        else:
            self.__state1.nsiteslist + self.__state2.nsiteslist

    @property
    def array(self):
        return np.concatenate((self.__state1.array,
                               self.__state2.array), axis=0)

    @property
    def nspace(self):
        return self.__nspace

    @property
    def nparticles(self):
        return self.__nparticles

    @property
    def nsites(self):
        return self.__nsites

    @property
    def index(self):
        return self.__index

    @property
    def parttype(self):
        return self.__parttype.copy()

    @property
    def max(self):
        return self.__max

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return repr(self.__state1) + "*" + repr(self.__state2)

    def __eq__(self, other):
        if (type(self) == type(other)):
            return (self.__state1 == other.__state1
                    and self.__state2 == other.__state2)

    def __mul__(self, other):
        return TensorProductState(self, other)

    def copy(self):
        return self.__class__(self.__state1, self.__state2)

    def deepcopy(self):
        return self.__class__(self.__state1.deepcopy(),
                              self.__state2.deepcopy())

    @staticmethod
    def same_space(elem1, elem2):
        if (type(elem1) == type(elem2)):
            return (elem1.__state1.same_space(elem1.state1, elem2.state1) and
                    elem1.__state2.same_space(elem1.state2, elem2.state2))
        else:
            return False

    def __calc_index(self):
        return self.__state1.index*self.__state2.max + self.__state2.index

    def __calc_max(self):
        return self.__state1.max*self.__state2.max

    def reset(self):
        self.__state1.reset()
        self.__state2.reset()
        self.__index = self.__calc_index()
        return self

    def vacuum(self):
        self.__state1.vacuum()
        self.__state2.vacuum()
        self.__index = self.__calc_index()
        return self

    def next_element(self):
        if (self.__state2.next_element()):
            self.__index = self.__calc_index()
            return True
        elif(self.__state1.next_element()):
            self.__state2.reset()
            self.__index = self.__calc_index()
            return True
        else:
            return False

    def range(self, elem1=None, elem2=None):
        if (elem2 is None):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                if (self.__class__.same_space(self, elem1)):
                    elem = self.deepcopy()
                    elem.reset()
                    if (elem.index < elem1.index):
                        while(not np.array_equal(elem.array, elem1.array)):
                            yield elem
                            elem.next_element()
                        return
                else:
                    raise ValueError("range can only end with"
                                     "elements of the same space")
        else:
            if (self.__class__.same_space(elem1, elem2)):
                if (elem1.index < elem2.index):
                    elem = elem1.deepcopy()
                    while(not np.array_equal(elem.array, elem2.array)):
                        yield elem
                        elem.next_element()
                    return
            else:
                raise ValueError("range can only be used between"
                                 "elements of the same space")

    def go_through(self, parttype):
        coeff2, elem2 = self.__state2.go_through(parttype)
        coeff1, elem1 = self.__state1.go_through(parttype)
        return (self.__class__(elem1, elem2), (coeff1*coeff2))

    def operator(self, op):
        parttype = self.__parttype[op[0]]
        ispace = op[0] + 1
        if (ispace > self.nspace or ispace < 1):
            raise ValueError("space number " + str((ispace-1))
                             + " does not exist in this product")
        if (ispace > self.__state1.nspace):
            elem, coeff = self.__state2.operator(
                [op[0]-self.__state1.nspace, op[1], op[2]])
            if (elem is None):
                return (None, 0)
            else:
                return (self.__class__(self.__state1, elem), coeff)
        else:
            elem2, coeff2 = self.__state2.go_through(parttype)
            elem1, coeff1 = self.__state1.operator(op)
            if (elem1 is None):
                return (None, 0)
            else:
                return (self.__class__(elem1, elem2), (coeff1*coeff2))


def apply(h_list, elem):
    outlist = []
    for i in h_list:
        val = i[0]
        telem = elem.deepcopy()
        coeff = 1
        for op in reversed(i[1]):
            telem, tcoeff = telem.operator(op)
            coeff = coeff*tcoeff
            if (telem is None):
                break
        if (telem is not None):
            outlist.append((telem.index, (coeff*val)))
    outlist.sort(key=lambda x: x[0])
    return outlist


def apply_to_ket(h_list, ketin, elemin):
    ket = np.zeros(elemin.max, dtype=complex)
    for elem in elemin.range():
        for i in h_list:
            val = i[0]
            telem = elem.deepcopy()
            coeff = ketin[elem.index]
            for op in reversed(i[1]):
                telem, tcoeff = telem.operator(op)
                coeff = coeff*tcoeff
                if (telem is None):
                    break
            if (telem is not None):
                ket[telem.index] += coeff*val
    return ket


def apply_to_ket_2spaces(h_list, ketin, elemin, elemout):
    ket = np.zeros(elemout.max, dtype=complex)
    for elem in elemin.range():
        for i in h_list:
            val = i[0]
            telem = elem.deepcopy()
            coeff = ketin[elem.index]
            for op in reversed(i[1]):
                telem, tcoeff = telem.operator(op)
                coeff = coeff*tcoeff
                if (telem is None):
                    break
            if (telem is not None):
                ket[telem.index] += coeff*val
    return ket


def uterm(u, ispace1, ispace2, ipos):
    return [[-u, [[ispace1, ipos, 1], [ispace2, ipos, 1],
            [ispace1, ipos, -1], [ispace2, ipos, -1]]]]


def tterm(t, ispace, ipos1, ipos2):
    return [[t, [[ispace, ipos1, 1], [ispace, ipos2, -1]]],
            [t, [[ispace, ipos2, 1], [ispace, ipos1, -1]]]]


def vterm(v, ispace, ipos):
    return [[v, [[ispace, ipos, 1], [ispace, ipos, -1]]]]


def chain(ut, tt, nsites, periodic=True):
    ht = []
    for i in range(nsites-1):
        ht += uterm(ut, 0, 1, i)
        ht += tterm(tt, 0, i, i+1)
        ht += tterm(tt, 1, i, i+1)
    ht += uterm(ut, 0, 1, nsites-1)
    if (periodic):
        ht += tterm(tt, 0, nsites-1, 0)
        ht += tterm(tt, 1, nsites-1, 0)
    return ht


def hlist_to_csr(h_list, elem):
    lrow = []
    lcol = []
    ldata = []
    for el in elem.range():
        tlist = apply(h_list, el)
        for ind, coeff in tlist:
            lcol.append(el.index)
            lrow.append(ind)
            ldata.append(coeff)
    row = np.array(lrow)
    col = np.array(lcol)
    data = np.array(ldata)
    return csr_matrix((data, (row, col)),
                      shape=(elem.max, elem.max), dtype=complex)


class SIndex:

    def __init__(self, nspace, nsiteslist):
        self.__nspace = nspace
        self.__nsiteslist = nsiteslist

    @classmethod
    def initstate(cls, state):
        return cls(state.nspace, state.nsiteslist)

    @property
    def nspace(self):
        return self.__nspace

    @property
    def nsiteslist(self):
        return self.__nsiteslist

    def v(self, ll):
        ispace = ll[0]
        ipos = ll[1]
        return sum(self.__nsiteslist[:ispace]) + ipos

    def vi(self, ind):
        ispace = 0
        acc = 0
        ipos = 0
        for i in self.__nsiteslist:
            if (ind >= (acc + i)):
                ispace = ispace + 1
                acc = acc + i
            else:
                ipos = ind - acc
                break
        return ispace, ipos


class MFHamiltonian:

    def __init__(self, nsiteslist, ha):
        self.__nsiteslist = nsiteslist
        self.__nsites = sum(nsiteslist)
        self.__ha = ha
        self.__sindex = SIndex(np.size(nsiteslist, 0), nsiteslist)

    @classmethod
    def initstate(cls, state, ha):
        return cls(state.nsiteslist, ha)

    def hamiltonian_matrix(self, densmat):
        hamat = np.zeros([self.__nsites, self.__nsites])
        for elem in self.__ha:
            val = elem[0]
            op = elem[1]
            if (len(op) == 2):
                i = self.__sindex.v(op[0])
                j = self.__sindex.v(op[1])
                hamat[i,j] += val
            elif (len(op) == 4):
                i = self.__sindex.v(op[0])
                j = self.__sindex.v(op[1])
                l = self.__sindex.v(op[2])
                k = self.__sindex.v(op[3])
                hamat[j,k] -= val*densmat[l,i]
                hamat[j,l] += val*densmat[k,i]
                hamat[i,k] += val*densmat[l,j]
                hamat[i,l] -= val*densmat[k,j]
        return hamat

    def sp_hamiltonian_matrix(self):
        hamat = np.zeros([self.__nsites, self.__nsites])
        for elem in self.__ha:
            val = elem[0]
            op = elem[1]
            if (len(op) == 2):
                i = self.__sindex.v(op[0])
                j = self.__sindex.v(op[1])
                hamat[i,j] += val
        return hamat

def densitymatrix(orbitals, occup):
    nsize = np.size(orbitals, 0)
    densmat = np.zeros([nsize,nsize])
    for l in range(np.size(orbitals,1)):
        densmat += occup[l]*np.outer(
            orbitals[:,l],(np.conjugate(orbitals[:,l])))
    return densmat

def HFenergy(ha,nsiteslist, orbitals, occup):
    densmat = densitymatrix(orbitals, occup)
    sindex = SIndex(np.size(nsiteslist), nsiteslist)
    energy = 0
    for elem in ha:
        val = elem[0]
        op = elem[1]
        if (len(op) == 2):
            i = sindex.v(op[0])
            j = sindex.v(op[1])
            energy += val*densmat[j,i]
        if (len(op) == 4):
            i = sindex.v(op[0])
            j = sindex.v(op[1])
            l = sindex.v(op[2])
            k = sindex.v(op[3])
            energy += val*densmat[l,j]*densmat[k,i]
            energy -= val*densmat[k,j]*densmat[l,i]
    return energy





def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

def solve_HF(ha, npart, nsiteslist, convergence, restricted=True, verbose=True):
    nsites = sum(nsiteslist)
    mfha = MFHamiltonian(nsiteslist,ha)
    occup = np.zeros(nsites)
    occup[:npart] = 1
    mfmat = mfha.sp_hamiltonian_matrix()
    eigval, orbitals = np.linalg.eigh(mfmat)
    idx = eigval.argsort()
    eigval = eigval[idx]
    orbitals = orbitals[:,idx]
    if (not restricted):
        orbitals = orbitals + 0.0001*np.random.rand(nsites,nsites)
        orbitals = gram_schmidt(orbitals)
    densmat = densitymatrix(orbitals,occup)
    mfmat = mfha.hamiltonian_matrix(densmat)
    criterium = np.linalg.norm(densmat-np.identity(nsites))
    i = 0
    while (criterium > convergence):
        eigval, orbitals = np.linalg.eigh(mfmat)
        idx = eigval.argsort()
        eigval = eigval[idx]
        orbitals = orbitals[:,idx]
        new_densmat = densitymatrix(orbitals,occup)
        criterium = np.linalg.norm(new_densmat-densmat)
        densmat = 0.01*new_densmat + 0.99*densmat
        mfmat = mfha.hamiltonian_matrix(densmat)
        if (i%100000 == 0 and verbose) :
            print('hf iteration, convergence:')
            print(criterium)
        i = i + 1
    return orbitals, eigval



import copy

class RHFHamiltonian:

    def __init__(self, ispace1, ispace2, innsiteslist, ha):
        nsiteslist = innsiteslist.copy()
        del nsiteslist[ispace2]
        self.__ispace1 = ispace1
        self.__ispace2 = ispace2
        self.__nsiteslist = nsiteslist
        self.__nsites = sum(nsiteslist)
        self.__ha = ha
        self.__sindex = SIndex(np.size(nsiteslist,0), nsiteslist)

    @classmethod
    def initstate(cls, ispace1, ispace2, state, ha):
        return cls(ispace1, ispace2, state.nsiteslist, ha)

    def hamiltonian_matrix(self, densmat):
        hamat = np.zeros([self.__nsites,self.__nsites])
        for elem in self.__ha:
            val = elem[0]
            op = elem[1]
            temp_op = copy.deepcopy(op)
            for telem in temp_op:
                if (telem[0] > self.__ispace2):
                        telem[0] -= 1

            if (len(op) == 2):
                if (op[0][0] != self.__ispace2 and op[1][0] != self.__ispace2):
                    i = self.__sindex.v(temp_op[0])
                    j = self.__sindex.v(temp_op[1])
                    hamat[i,j] += val

            elif (len(op) == 4):

                for telem in temp_op:
                    if (telem[0] == self.__ispace2):
                            telem[0] = self.__ispace1
                i = self.__sindex.v(temp_op[0])
                j = self.__sindex.v(temp_op[1])
                l = self.__sindex.v(temp_op[2])
                k = self.__sindex.v(temp_op[3])
                if not (op[0][0] == self.__ispace2
                        ^ op[2][0] == self.__ispace2):
                    if (op[1][0] != self.__ispace2
                        and op[3][0] != self.__ispace2):
                        hamat[j,k] -= val*densmat[l,i]
                if not (op[0][0] == self.__ispace2
                        ^ op[3][0] == self.__ispace2):
                    if (op[1][0] != self.__ispace2
                        and op[2][0] != self.__ispace2):
                        hamat[j,l] += val*densmat[k,i]
                if not (op[1][0] == self.__ispace2
                        ^ op[2][0] == self.__ispace2):
                    if (op[0][0] != self.__ispace2
                        and op[3][0] != self.__ispace2):
                        hamat[i,k] += val*densmat[l,j]
                if not (op[1][0] == self.__ispace2
                        ^ op[3][0] == self.__ispace2):
                    if (op[0][0] != self.__ispace2
                        and op[2][0] != self.__ispace2):
                        hamat[i,l] -= val*densmat[k,j]
        return hamat

    def sp_hamiltonian_matrix(self):
        hamat = np.zeros([self.__nsites, self.__nsites])
        for elem in self.__ha:
            val = elem[0]
            op = elem[1]
            temp_op = op.copy()
            for elem in temp_op:
                if (elem[0] > self.__ispace2):
                        elem[0] -= 1
            if (len(op) == 2):
                if (op[0][0] != self.__ispace2 and op[1][0] != self.__ispace2):
                    i = self.__sindex.v(temp_op[0])
                    j = self.__sindex.v(temp_op[1])
                    hamat[i,j] += val
        return hamat


def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

def convertback_orbeig(orbeig):
    eig = orbeig[1]
    orb = orbeig[0]
    nsites = np.size(eig,0)
    eigval = np.zeros(2*nsites)
    orbitals = np.zeros([2*nsites,2*nsites])
    for j in range(nsites):
        orbitals[:nsites,2*j] = orb[:,j]
        eigval[2*j] = eig[j]
        orbitals[nsites:,2*j+1] = orb[:,j]
        eigval[2*j+1] = eig[j]
    idx = eigval.argsort()
    eigval = eigval[idx]
    orbitals = orbitals[:,idx]
    return orbitals, eigval

def solve_RHF(ha, npart, nsiteslist, convergence, ispace1 = 0,
              ispace2 = 1, verbose=True):
    nsites = sum(nsiteslist)-nsiteslist[ispace2]
    mfha = RHFHamiltonian(ispace1,ispace2,nsiteslist,ha)
    occup = np.zeros(nsites)
    occup[:npart//2] = 1
    if (npart%2 == 1):
        occup[:npart//2+1] = 1
    mfmat = mfha.sp_hamiltonian_matrix()
    eigval, orbitals = np.linalg.eigh(mfmat)
    idx = eigval.argsort()
    eigval = eigval[idx]
    orbitals = orbitals[:,idx]
    densmat = densitymatrix(orbitals,occup)
    mfmat = mfha.hamiltonian_matrix(densmat)
    criterium = np.linalg.norm(densmat-np.identity(nsites))
    i = 0
    while (criterium > convergence):
        eigval, orbitals = np.linalg.eigh(mfmat)
        idx = eigval.argsort()
        eigval = eigval[idx]
        orbitals = orbitals[:,idx]
        new_densmat = densitymatrix(orbitals,occup)
        criterium = np.linalg.norm(new_densmat-densmat)
        densmat = 0.01*new_densmat + 0.99*densmat
        mfmat = mfha.hamiltonian_matrix(densmat)
        if (i%100000 == 0 and verbose) :
            print('hf iteration, convergence:')
            print(criterium)
        i = i + 1
    return convertback_orbeig((orbitals, eigval))


def orb_to_ket(orbitals, occup, elem, restricted=True):
    npart = elem.nparticles
    nsites = elem.nsites
    norbs = np.size(orbitals,1)

    if restricted:
        det = elem.deepcopy()
    else:
        BasicFermions.firstdet(npart,nsites)
    ket = np.zeros(det.max)

    det.vacuum()
    sind = SIndex(det.nspace,det.nsiteslist)

    def detrec(npart, norbs, orbitals, occup, val, det):
        l = 0
        while (l < norbs and round(occup[l]) == 0):
            l = l + 1
        if (l < norbs):
            for i in range(nsites):
                ispace, ipos = sind.vi(i)
                det1, sign = det.operator([ispace,ipos,1])
                if (det1 != None):
                    detrec(npart-1,norbs-l-1,orbitals[:,l+1:],
                            occup[l+1:],(sign*val*orbitals[i,l]),det1)
        else:
            k = det.index
            ket[k] += val

    detrec(npart,norbs,orbitals,occup,1,det)

    return ket



def psi_timestep(Psi, ha, dt) :
    enerlist, eigvec = np.linalg.eig(ha)
    enerlist = enerlist - min(enerlist)
    Psi_new = np.dot(
        eigvec@np.diag(np.exp(-1j*(enerlist)*dt))@np.linalg.inv(eigvec),Psi)
    return Psi_new


def propag(ket, tmax, dt, PsiInit, hlist, vlist, vamplitude, tprint):

    h0 = hlist_to_csr(hlist, ket).toarray()
    v = hlist_to_csr(vlist, ket).toarray()

    tarray = np.arange(0, tmax, dt)
    tplist = []
    Psilist = []

    iprint = 0
    Psi = PsiInit

    for it, tt in enumerate(tarray):
        if (abs(tt-iprint*tprint) <= dt/2):
            tplist.append(tt)
            Psilist.append(Psi)
            iprint = iprint + 1
        ht = h0 + vamplitude(tt)*v
        Psi = psi_timestep(Psi, ht, dt)

    return np.array(tplist), np.array(Psilist).transpose()

def propag_ham(ket, tmax, dt, PsiInit, h0, v, vamplitude, tprint):

    tarray = np.arange(0, tmax, dt)
    tplist = []
    Psilist = []

    iprint = 0
    Psi = PsiInit

    for it, tt in enumerate(tarray):
        if (abs(tt-iprint*tprint) <= dt/2):
            tplist.append(tt)
            Psilist.append(Psi)
            iprint = iprint + 1
        ht = h0 + vamplitude(tt)*v
        Psi = psi_timestep(Psi, ht, dt)

    return np.array(tplist), np.array(Psilist).transpose()

def propag_ham_free(ket, tmax, dt, PsiInit, h0, tprint):

    tarray = np.arange(0, tmax, dt)
    tplist = []
    Psilist = []

    enerlist, eigvec = np.linalg.eig(h0)
    enerlist = enerlist - min(enerlist)

    iprint = 0
    Psi = PsiInit

    for it, tt in enumerate(tarray):
        if (abs(tt-iprint*tprint) <= dt/2):
            tplist.append(tt)
            Psilist.append(Psi)
            iprint = iprint + 1
        Psi = np.dot(
            eigvec@np.diag(np.exp(-1j*(enerlist)*dt))@np.linalg.inv(eigvec),Psi)

    return np.array(tplist), np.array(Psilist).transpose()


def propag_dec(ket, tmax, dt, PsiInit, hlist, vlist, vamplitude, tprint, declist):

    h0 = hlist_to_csr(hlist, ket).toarray()
    v = hlist_to_csr(vlist, ket).toarray()
    bb = hlist_to_csr(declist, ket).toarray()

    tarray = np.arange(0, tmax, dt)
    tplist = []
    Psilist = []

    iprint = 0
    Psi = PsiInit

    for it, tt in enumerate(tarray):
        if (abs(tt-iprint*tprint) <= dt/2):
            tplist.append(tt)
            Psilist.append(Psi)
            iprint = iprint + 1
        ht = h0 + vamplitude(tt)*v
        Psi = psi_timestep(Psi, ht, dt)
        Psi = psi_timestep(Psi, bb, -1j*dt)
        Psi = Psi / np.linalg.norm(Psi)
    return np.array(tplist), np.array(Psilist).transpose()



# If not imported run an example
if __name__ == "__main__":
    sdet = BasicFermions.firstdet(1,2)
    det = sdet*sdet

    U = 4
    ############# exact hamiltonian #############
    ht = chain(U,-1/2,2,periodic=False)
    ht= ht + vterm(1.5,0,0)
    ht= ht + vterm(1.5,1,0)
    ht= ht + vterm(-1.5,0,1)
    ht= ht + vterm(-1.5,1,1)
    # transforms into a csr format
    ht_csr = hlist_to_csr(ht,det)
    print(ht_csr.toarray())
    val = eigs(ht_csr,k=1,which='SR',return_eigenvectors=False)
    print(val)

    ############# HF hamiltonian #############
    orb, eigval = solve_HF(ht,2,[2,2],0.00000001)
    print(orb)
    print(eigval)
    occup = np.zeros(np.size(orb,0))
    occup[:2] = 1

    # HF energy using density matrix
    hfenergy = HFenergy(ht,det.nsiteslist,orb,occup)
    print(hfenergy)

    # HF energy computing from ket (works only for restricted HF)
    ket = orb_to_ket(orb,occup,det)
    print(ket)
    print(np.dot(ket,ket))
    print(np.dot(ket,apply_to_ket(ht,ket,det)))



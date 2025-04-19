#!/usr/bin/env python
#coding: utf-8


from scipy.special import binom
from scipy.sparse import csr_matrix
import numpy as np

np.set_printoptions(linewidth=120)


class BasicFermions:
    """
    Stores fermionic slater determinant in the form of a 1/0 vector
    This class contains the rules of the Hilbert space to which the determinant belongs
    Each element (determinant) of a Hilbert space possesses its unique index 

    Attributes:
        nsites : int
            number of sites (or single particle wavefunctions)
        nsiteslist : int
            list of the number of sites of each Hilbert space
            only returns [nsites] if the object is not a tensorproduct
        nparticles : int 
            number of particles in this determinant
        nconserved : bool
            tells if the number of particle is conserved
            True: Hilbert space with 'nparticles' particles / False: Fock space
        array : int
            array of 1 or 0 that represents the determinant
        list : int
            list of 1 or 0 that represents the determinant
        nspace : int
            number of sub-Hilbert space (always 1 for this class)
        max : int 
            size of the space considered
        index : int
            return the unique determinant index in the considered space
        parttype : str 
            name of the particle (important for commutation rules)
        comrules : dict
            name of the particle associated with the phase of the
            commutation
    
    """

    def __init__(self, array, nconserved=True, parttype='fermions',
                 comrules={'fermions': -1}):
        """
        Constructor of the BasicFermions object

        Parameters:
            array : int
                initial configuration of the determinant
            nconserved : bool
                if True(default) the number of particles is
                conserved
            parttype : str
                name of the particle (for commutation rules)
            comrules : dict 
                name of the particle associated with the phase
                of the commutation
        """
        self.__nconserved = nconserved
        self.__nspace = 1
        self.__parttype = parttype
        self.__comrules = comrules
        self.array = np.array(array)
        self.array.flags.writeable = False

    @classmethod
    def firstdet(cls, nparticles, nsites, nconserved=True, parttype='fermions',
                 comrules={'fermions': -1}):
        """
        Alternate constructor + initialization of the BasicFermions object

        Parameters:
            nparticles : int
                number of particles (ignored if nconserved=False)
            nsites : int
                number of sites
            nconserved : bool
                if True(default) the number of particles is conserved
            parttype : str
                name of the particle (for commutation rules)
            comrules : dict
                name of the particle associated with the phase
                of the commutation
        """
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
        """
        You can change the determinant by hand (the array of 0s and 1s)
        but it may change the Hilbert space 
        USE CAUTIOUSLY
        """
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
        """
        Equality boolean operator
        Returns True if two elements are the same in the same Hilbert space
        (same array, same rules)
        """
        if (type(self) == type(other)):
            return (np.array_equal(self.array, other.array)
                    and self.nconserved == other.nconserved
                    and self.parttype == other.parttype
                    and self.comrules == other.comrules)
        else:
            return False

    def __mul__(self, other):
        return TensorProductState(self, other)

    def __getitem__(self, ispace):        
        if (ispace != 0):
            raise ValueError("space number " + str((ispace))
                             + " does not exist")
        return self

    def copy(self):
        return self.__class__(self.array, self.nconserved)

    def deepcopy(self):
        return self.__class__(self.array, self.nconserved)

    @staticmethod
    def same_space(elem1, elem2):
        """
        Returns True if two elements are in the same Hilbert space
        They do not need to have the same configuration (i.e. same array)
        """
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
        """
        Computes the size of the Hilbert space
        """
        if (self.nconserved):
            return int(binom(self.nsites, self.nparticles))
        else:
            return 2**self.nsites

    def reset(self):
        """
        Sets the element to the first element in the Hilbert space
        """
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
        """
        Sets the element to the vacuum state
        """
        self.array = np.zeros(self.nsites, dtype=int)
        self.__nparticles = 0
        self.__calc_index()
        return self

    def next_element(self):
        """
        Transforms the current element into the next one the Hilbert space
        with the index = index + 1 if it exists
        Returns True if the next element exists
        Returns False and leaves the element unchanged if the next element
        does not exist
        """
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
        """
        Iterator over the elements of the Hilbert space

        Arguments:
        *If no element is given, it does the full loop starting from
            the first element (the only actually useful case)
        *If one element is given, then it will go from the first element
            to this element NOT INCLUDED
        *If two elements (elem1, elem2) are given, then it will start from elem1 and end
            at elem2 NOT INCLUDED
        To start at elem1 and go to the end of the Hilbert space (last element included)
        set elem2 to "end"
        """
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
        elif (elem2 == "end"):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                elem = elem1.deepcopy()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
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
        """
        Computes the sign or coefficient created by an operator that acts on another
        Hilbert space when it commutes with (or goes through) the product of operators
        defining this determinant
        Returns a copy of this element and the coefficient(real number)
        in the form (elem, coeff)
        """
        if (parttype in self.__comrules):
            coeff = (self.__comrules[parttype])**self.__nparticles
            return (self.copy(), coeff)
        else:
            coeff = 1
            return (self.copy(), coeff)

    def operator(self, op):
        """
        Computes how a given operator acts on this element
        The operator is in the form of a list [ind, 1] (for creation operator)
        or [ind, -1] (for annihilation operator) where ind is the index of the site
        where the particle is created/annihilated
        Returns (elem, coeff) where  elem is the modified element 
        (not necessarily in the same Hilbert space) and coeff 
        is the corresponding coefficient created in front of the determinant
        by the action of the operator.
        When this action returns 0 (like annihilating a 0), it returns (None, 0)
        """
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
    """
    Stores bosonic permanent in the form of a vector of occupation numbers
    This class contains the rules of the Hilbert space to which the permanent belongs
    Each element of a given Hilbert space possesses its unique index 

    Attributes:
        nparticles : int 
            number of particles in this permanent
        nsites : int
            number of sites/modes (or single particle wavefunctions)
        nmax : int
            array indicating the maximum number of bosons allowed in each site/mode
        ntotmax : int
            maximum number of bosons allowed over all the modes
        ntotlimited : bool
            If True, the TOTAL number of bosons is limited to ntotmax
            If False, each mode has its own limit of the number of bosons it
            can accept (see the array nmax), independent of the other modes
        array : int
            array of occupation numbers that represents the permanent
        list : int
            list of occupation numbers that represents the permanent
        nsiteslist : int
            list of the number of sites/modes of each Hilbert space
            only returns [nsites] if the object is not a tensorproduct
        nspace : int
            number of sub-Hilbert space (always 1 for this class)
        max : int 
            size of the space considered
        index : int
            return the unique index of the element in in the considered space
        parttype : str 
            name of the particle (important for commutation rules)
        comrules : dict
            name of the particle associated with the phase of the
            commutation
    
    """

    def __init__(self, array, nmax, ntotlimited=False, ntotmax=0,
                 parttype='bosons', comrules={'bosons': 1}):
        """
        Constructor of the BasicBosons object

        Parameters:
            array : int
                initial configuration of the permanent
            nmax : int
                array indicating the maximum number of bosons allowed in each site/mode
            ntotlimited : bool
                if True, only the total number of bosons over all modes is limited to ntotmax
                and nmax is overwritten
                if False, each mode has an independent limit set by nmax 
                default is False
            ntotmax : int
                only used if ntotlimited == True, sets the maximum number of bosons
                over all modes
                default is 0
            parttype : str
                name of the particle (for commutation rules)
            comrules : dict 
                name of the particle associated with the phase
                of the commutation
        """
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
        """
        Alternate constructor + initialization of the BasicBosons object

        For a BasicBosons object, the initial array is an array of zeros

        Parameters:
            nsites : int
                number of sites/modes
            nmax : int
                array indicating the maximum number of bosons allowed in each site/mode
            ntotlimited : bool
                if True, only the total number of bosons over all modes is limited and
                nmax is overwritten
                if False, each mode has an independent limit set by nmax 
                default is False
            ntotmax : int
                only used if ntotlimited == True, sets the maximum number of bosons
                over all modes
                default is 0
            parttype : str
                name of the particle (for commutation rules)
            comrules : dict 
                name of the particle associated with the phase
                of the commutation
        """
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
        """
        You can change the permanent by hand (the array of particle numbers) 
        but it may change the Hilbert space.
        If will not accept an array that does not respect the rules for nmax or ntotmax
        USE CAUTIOUSLY
        """
        if (self.ntotlimited):
            if (type(array) == np.ndarray and
                    np.sum(array) <= self.ntotmax):
                self.__array = array
                self.__nsites = self.__array.size
                self.__max = self.__calc_max()
                self.__index = self.__calc_index()
                self.__nmax = self.ntotmax*np.ones_like(array)
            else:
                raise ValueError("only numpy arrays of integers with sum<ntotmax are accepted")
        else:
            if (type(array) == np.ndarray and
                    array.size == (self.nmax).size and
                    np.prod(np.less_equal(array, self.nmax))):
                self.__array = array
                self.__nsites = self.__array.size
                self.__max = self.__calc_max()
                self.__index = self.__calc_index()
            else:
                raise ValueError("only numpy arrays of integers <nmax (and same size) are accepted")

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return (str(self.__class__.__name__) + "(" + str(self.list) +
                ", " + np.array2string(self.nmax, separator=', ') +
                ", " + str(self.ntotlimited) + ", " + str(self.ntotmax) +
                ")")

    def __eq__(self, other):
        """
        Equality boolean operator
        Returns True if two elements are the same in the same Hilbert space
        (same array, same rules)
        """
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

    def __getitem__(self, ispace):        
        if (ispace != 0):
            raise ValueError("space number " + str((ispace))
                             + " does not exist")
        return self

    def copy(self):
        return self.__class__(self.array, self.nmax,
                              self.ntotlimited, self.ntotmax)

    def deepcopy(self):
        return self.__class__(self.array, self.nmax,
                              self.ntotlimited, self.ntotmax)

    @staticmethod
    def same_space(elem1, elem2):
        """
        Returns True if two elements are in the same Hilbert space
        They do not need to have the same configuration (i.e. same array)
        """
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
        """
        Computes the size of the Hilbert space
        """
        if (self.ntotlimited):
            return int(binom(self.ntotmax+self.nsites, self.ntotmax))
        else:
            return np.prod(self.nmax+1)

    def reset(self):
        """
        Sets the element to the first element in the Hilbert space
        """
        self.array = np.zeros(self.nsites, dtype=int)
        self.__calc_index()
        return self

    def vacuum(self):
        """
        Sets the element to the vacuum state
        """
        self.array = np.zeros(self.nsites, dtype=int)
        self.__calc_index()
        return self

    def next_element(self):
        """
        Transforms the current element into the next one the Hilbert space
        with the index = index + 1 if it exists
        Returns True if the next element exists
        Returns False and leaves the element unchanged if the next element
        does not exist
        """
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
        """
        Iterator over the elements of the Hilbert space

        Arguments:
        *If no element is given, it does the full loop starting from
            the first element (the only actually useful case)
        *If one element is given, then it will go from the first element
            to this element NOT INCLUDED
        *If two elements (elem1, elem2) are given, then it will start from elem1 and end
            at elem2 NOT INCLUDED
        To start at elem1 and go to the end of the Hilbert space (last element included)
        set elem2 to "end"
        """
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
        elif (elem2 == "end"):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                elem = elem1.deepcopy()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
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
        """
        Computes the sign or coefficient created by an operator that acts on another
        Hilbert space when it commutes with (or goes through) the product of operators
        defining this state
        Returns a copy of this element and the coefficient(real number)
        in the form (elem, coeff)
        """
        coeff = 1
        return (self.copy(), coeff)

    def operator(self, op):
        """
        Computes how a given operator acts on this element
        The operator is in the form of a list [ind, 1] (for creation operator)
        or [ind, -1] (for annihilation operator) where ind is the index of the site
        where the particle is created/annihilated
        Returns (elem, coeff) where  elem is the modified element 
        (not necessarily in the same Hilbert space) and coeff 
        is the corresponding coefficient created in front of the determinant
        by the action of the operator.
        When this action returns 0 (like annihilating a 0), it returns (None, 0)
        """
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
    """
    Stores tensor product states in the form of a recursive list of states
    It is created as a the result of a product between 
    two Hilbert spaces objects (BasicBosons or BasicFermions)
    or TensorProductState objects
    It combines the properties of the Hilbert spaces it is the product of
    Each element of this Hilbert space possesses its unique index

    Attributes:
        state1 : object
            State/Hilbert space on the left of the product sign
            Either of Basic type or a TensorProductState
        state2 : object
            State/Hilbert space on the right of the product sign
            Always of Basic type
            States are stored as
            state1*state2 = ((((..*state)*state)*state)*state2
        list : int 
            list of lists
            each sublist is the occupation numbers of each 
            sub Hilbert space
        nsiteslist : int
            list of the number of sites of each sub Hilbert space
        array : int
            array of the occupation numbers of each site/modes
            no separation is visible between the different Hilbert spaces
            see nsiteslist to know which position corresponds to which
            Hilbert space
        nspace : int
            number of sub-Hilbert spaces in the product
        nparticles : int
            total number of particles
        nsites : int
            total number of sites+modes
        index : int
            return the unique index of the element in in the considered space
        parttype : str 
            list of the names of the particles
        max : int 
            size of the product space    
    """

    def __init__(self, state1, state2):
        """
        Constructor of the TensorProductState object
        as a product of state1 and state2
        """
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
        """
        Equality boolean operator
        Returns True if two elements are the same in the same Hilbert space
        (same array, same rules)
        """
        if (type(self) == type(other)):
            return (self.__state1 == other.__state1
                    and self.__state2 == other.__state2)

    def __mul__(self, other):
        return TensorProductState(self, other)

    def __getitem__(self, ispace):
        if (ispace >= self.nspace or ispace < 0):
            raise ValueError("space number " + str((ispace))
                             + " does not exist in this product")
        if (ispace >= self.__state1.nspace):
            return self.__state2[ispace-self.__state1.nspace]
        else:
            return self.__state1[ispace]

    def copy(self):
        return self.__class__(self.__state1, self.__state2)

    def deepcopy(self):
        return self.__class__(self.__state1.deepcopy(),
                              self.__state2.deepcopy())

    @staticmethod
    def same_space(elem1, elem2):
        """
        Returns True if two elements are in the same Hilbert space
        They do not need to have the same configuration (i.e. same array)
        """
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
        """
        Sets the element to the first element in the Hilbert space
        """
        self.__state1.reset()
        self.__state2.reset()
        self.__index = self.__calc_index()
        return self

    def vacuum(self):
        """
        Sets the element to the vacuum state
        """
        self.__state1.vacuum()
        self.__state2.vacuum()
        self.__index = self.__calc_index()
        return self

    def next_element(self):
        """
        Transforms the current element into the next one the Hilbert space
        with the index = index + 1 if it exists
        Returns True if the next element exists
        Returns False and leaves the element unchanged if the next element
        does not exist
        """
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
        """
        Iterator over the elements of the Hilbert space

        Arguments:
        *If no element is given, it does the full loop starting from
            the first element (the only actually useful case)
        *If one element is given, then it will go from the first element
            to this element NOT INCLUDED
        *If two elements (elem1, elem2) are given, then it will start from elem1 and end
            at elem2 NOT INCLUDED
        To start at elem1 and go to the end of the Hilbert space (last element included)
        set elem2 to "end"
        """
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
        elif (elem2 == "end"):
            if (elem1 is None):
                elem = self.deepcopy()
                elem.reset()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
            else:
                elem = elem1.deepcopy()
                exist = True
                while(exist):
                    yield elem
                    exist = elem.next_element()
                return
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
        """
        Computes the sign or coefficient created by an operator that acts on another
        Hilbert space when it commutes with (or goes through) the product of operators
        defining this state
        Returns a copy of this element and the coefficient(real number)
        in the form (elem, coeff)
        """
        coeff2, elem2 = self.__state2.go_through(parttype)
        coeff1, elem1 = self.__state1.go_through(parttype)
        return (self.__class__(elem1, elem2), (coeff1*coeff2))

    def operator(self, op):
        """
        Computes how a given operator acts on this element
        The operator is in the form of a list [ind, 1] (for creation operator)
        or [ind, -1] (for annihilation operator) where ind is the index of the site
        where the particle is created/annihilated
        Returns (elem, coeff) where  elem is the modified element 
        (not necessarily in the same Hilbert space) and coeff 
        is the corresponding coefficient created in front of the determinant
        by the action of the operator.
        When this action returns 0 (like annihilating a 0), it returns (None, 0)
        """
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



# If not imported run an example
if __name__ == "__main__":
    from scipy.sparse.linalg import eigs

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


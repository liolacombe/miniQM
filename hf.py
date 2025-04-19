#!/usr/bin/env python
#coding: utf-8

import numpy as np

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

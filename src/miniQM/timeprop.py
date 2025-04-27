# !/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg

def lanczos_timestep(ketin, hmat, dt, nkrylov):
    avec = np.zeros(nkrylov)
    bvec = np.zeros(nkrylov-1)
    vp = ketin
    vn = hmat@vp
    an = np.real(vn.conj().T@vp)
    avec[0] = an
    vn = vn - an*vp
    bn = np.linalg.norm(vn)
    vn = vn/bn
    for ik in range(nkrylov-1):
        bvec[ik] = bn
        w = hmat@vn
        an = np.real(vn.conj().T@w)
        avec[ik+1] = an
        w = w - an*vn - bn*vp
        vp = vn
        bn = np.linalg.norm(w)
        vn = w/bn
    # diagonalize in krylov space
    kener, keigvec = scipy.linalg.eigh_tridiagonal(avec, bvec, select='a')
    # ketin is [1,0,0...] in the krylov basis
    kv = keigvec[0,:].conj()
    # so kv is now P inverse times ketin
    kv = keigvec@(np.exp(-1j*kener*dt)*kv)
    # kv is now P@e^(-iHdt)@P_inv@ 
    vp = ketin
    ketout = kv[0]*vp
    vn = hmat@vp
    an = np.real(vn.conj().T@vp)
    vn = vn - an*vp
    bn = np.linalg.norm(vn)
    vn = vn/bn
    for ik in range(nkrylov-1):
        ketout += kv[ik+1]*vn
        w = hmat@vn
        an = np.real(vn.conj().T@w)
        w = w - an*vn - bn*vp
        vp = vn
        bn = np.linalg.norm(w)
        vn = w/bn
    return ketout
    

def psi_timestep(Psi, ha, dt):
    enerlist, eigvec = np.linalg.eig(ha)
    enerlist = enerlist - min(enerlist)
    Psi_new = np.dot(
        eigvec@np.diag(np.exp(-1j*(enerlist)*dt))@np.linalg.inv(eigvec),Psi)
    return Psi_new


def propag(ket, tmax, dt, PsiInit, hlist, vlist, vamplitude, tprint):
    h0 = oplist_to_csr(hlist, ket).toarray()
    v = oplist_to_csr(vlist, ket).toarray()

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

    h0 = oplist_to_csr(hlist, ket).toarray()
    v = oplist_to_csr(vlist, ket).toarray()
    bb = oplist_to_csr(declist, ket).toarray()

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



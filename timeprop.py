#!/usr/bin/env python
#coding: utf-8

import numpy as np

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



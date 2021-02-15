# OBEpot and TwoBody Class taken from jupyter notebook of lecture 7


# for simplicity, we define the OBE exchange by numerical integration 
# and add one contact term to be determined using either a bound state or a scattering length 

#CORRECTED version

import numpy as np
import math as m
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
import scipy.special
from scipy.special import legendre
from scipy import interpolate
from scipy.optimize import curve_fit


class OBEpot:
    """Provides a method for the partial wave representation of the OBE potential. 
    
       The matrix elements are obtained by numerical intergration.
       The mass of the exchanged boson, the strength of the 
       interaction and the couter term is given on initialization. 
       The interaction is regularized using a cutoff that is also 
       given on init.
    """
    
    # this are common parameters for all instances 
    hbarc=197.327
    
    # init interaction
    def __init__(self, cutoff=500.0, C0=1.0, nx=12,mpi=138.0,A=-1.0):
        """Defines the one boson exchange for a given regulator, coupling strength and short distance parameter
        
        Parameters:
        cutoff -- regulator in MeV
        C0 -- strength of the short distance counter term (in s-wave) 
        A -- strength of OBE
        nx -- number of angular grid points for numerical integration
        mpi -- mass of exchange boson in MeV"""
        
        self.mpi = mpi/self.hbarc
        self.cutoff = cutoff/self.hbarc
        self.C0=C0
        self.A=A
        self.nx=nx
        
        self.xp=np.empty((self.nx),dtype=np.double)
        self.xw=np.empty((self.nx),dtype=np.double)
        self.xp,self.xw=leggauss(self.nx)
    
    
    
    # function defines the x integral 
    def _g(self,pp,p,k):
        """Calculates g function of the partial wave decomposition of OBE. 
        
           pp -- outgoing momentum 
           p -- incoming momentum
           k -- angular momentum"""
        
        # define prefact 
        # get the corresponding legendre polynomial 
        Pk = legendre(k)
        # define momentum transfer dependent on angles 
        qval=np.sqrt(p**2+pp**2-2*p*pp*self.xp)
        
        # build integral of regularized OBE 
        return float(np.sum(Pk(self.xp)/((qval**2+self.mpi**2))*self.xw*np.exp(-(qval**2+self.mpi**2)/self.cutoff**2)))
        
    # determines complete, regularized interaction     
    def v(self,pp,p,l):
        """Potential matrix element in fm**2
        
           pp -- outgoing momentum in fm**-1
           p -- incoming momentum in fm**-1
           l -- angular momentum""" 
        
        # first overall prefact of 1pi exchange part  (cancel 2pi factors!)
        prefact=self.A
        
        mat=prefact*self._g(pp,p,l)

        if (l==0):   # add s-wave counter term 
          mat+=self.C0*np.exp(-(pp**2+p**2)/self.cutoff**2)  # 4pi is take into account by spherical harmonics for l=0
                    
        return mat



class TwoBody:
    """Methods to obtain eigenvalues and eigenvectors for the bound state problem and for searches of the binding energy."""
    # define hbarc for unit conversion 
    hbarc=197.327  
    
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, mred=938.92/2,l=0,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            np1four=200,np2four=100):
        """Initialization of two-body solver. 
        
           The initialization defines the momentum grids and the interaction and partial wave to be used. 
           At this time, also the grid for Fourier transformation and for the Fourier transformed 
           wave function is given. 
           
           Parameters:
           pot -- object that defines the potential matrix elements (e.g. of class OBEpot).
           np1 -- number of grid points in interval [0,pb] 
           np2 -- number of grid points in interval [pb,pc]
           pa  -- half of np1 points are in interval [0,pa]
           pb  -- interval boundary as defined above 
           pc  -- upper integration boundary for the solution of the integral equation 
           mred -- reduces mass of the two bosons in MeV
           
           nr1 -- number of r points in interval [0,rb] 
           nr2 -- number of r points in interval [rb,rc]
           ra  -- half of np1 points are in interval [0,pa]
           rb  -- interval boundary as defined above 
           rc  -- upper integration boundary for the solution of the integral equation 
           
           np1four -- number of p points in interval [0,pb] for Fourier trafo
           np2four -- number of p points in interval [pb,pc] for Fourier trafo"""
        
        # store parameters (if necessary convert to fm)
        self.np1 = np1
        self.np2 = np2
        self.npoints  = np1+np2 
        self.mred=mred/self.hbarc
        self.pa=pa
        self.pb=pb
        self.pc=pc
        self.l=l 

        self.nr1 = nr1
        self.nr2 = nr2
        self.nrpoints  = nr1+nr2 
        self.ra=ra
        self.rb=rb
        self.rc=rc

        self.np1four = np1four
        self.np2four = np2four
        self.npfour  = np1four+np2four 

        # store grid points and weights for integral equations
        self.pgrid,self.pweight = self._trns(self.np1,self.np2,self.pa,self.pb,self.pc)
 
        # store grid points and weights for r space wave functions
        self.rgrid,self.rweight = self._trns(self.nr1,self.nr2,self.ra,self.rb,self.rc)
        
        # store grid points and weights for Fourier trafo 
        self.pfourgrid,self.pfourweight = self._trns(self.np1four,self.np2four,self.pa,self.pb,self.pc)
        
        # store underlying interaction
        self.pot=pot
        
    def _trns(self,np1,np2,pa,pb,pc):
      """Auxilliary method that provides transformed Gaus-Legendre grid points and integration weights.
      
         This is using a hyperbolic trafo shown in the lecture. 
         Parameter: 
         np1 --  grid points in ]0,pb[
         np2 --  grid points are distributed in ]pb,pc[ using a linear trafo
         
         pa  -- half of np1 points are in interval [0,pa]
         pb  -- interval boundary as defined above 
         pc  -- upper integration boundary """ 
    
      x1grid,x1weight=leggauss(np1)
      x2grid,x2weight=leggauss(np2)

      # trafo (1.+X) / (1./P1-(1./P1-2./P2)*X) for first interval 
      p1grid=(1.+x1grid) / (1./pa-(1./pa-2./pb)*x1grid)
      p1weight=(2.0/pa-2.0/pb)*x1weight / (1./pa-(1./pa-2./pb)*x1grid)**2

      # linear trafo 
      p2grid=(pc+pb)/2.0 + (pc-pb)/2.0*x2grid
      p2weight=(pc-pb)/2.0*x2weight
   
      pgrid=np.empty((self.npoints),dtype=np.double)
      pweight=np.empty((self.npoints),dtype=np.double)
    
      pgrid = np.concatenate((p1grid, p2grid), axis=None)
      pweight = np.concatenate((p1weight, p2weight), axis=None)
   
      return pgrid,pweight 

# set up set of equations and calculate eigenvalues 

    def eigv(self,E,neigv):
      """Solve two-body integral equation and return n-th eigenvalue, momentum grid and wave function. 

         Parameters:
         E -- energy used in the integral equation in fm**-1 
         neigv -- number of the eigenvalue to be used"""
   
    # set up the matrix amat for which eigenvalues have to be calculated 
      amat=np.empty((self.npoints,self.npoints),dtype=np.double)
      for i in range(self.npoints):
        for j in range(self.npoints): 
          amat[i,j]=np.real(1.0/(E-self.pgrid[i]**2/(2*self.mred))*self.pot.v(self.pgrid[i],self.pgrid[j],self.l)*self.pweight[j]*self.pgrid[j]**2)

    # determine eigenvalues using numpy's eig method        
      evalue,evec=np.linalg.eig(amat)
    
    # I now assume that the relevant eigenvalues are real to avoid complex arithmetic 
      evalue=np.real(evalue)
        
    # remove neigv-1 largest eigenvalues 
      for n in range(neigv-1):
        maxpos=np.argmax(evalue)
        evalue[maxpos]=0.0
    
    # take the next one 
      maxpos=np.argmax(evalue)
      eigv=evalue[maxpos]
    # define solution as unnormalized wave function 
      wf=evec[:,maxpos]
    # and normalize 
      norm=np.sum(wf**2*self.pweight[0:self.npoints]*self.pgrid[0:self.npoints]**2)
      wf=1/np.sqrt(norm)*wf
    
      return eigv,self.pgrid[0:self.npoints],self.pweight[0:self.npoints], wf

    
    def esearch(self,neigv=1,e1=-0.01,e2=-0.0105,elow=0.0,tol=1e-8):
        """Perform search for energy using the secant method. 
        
           Parameters:
           neigv -- number of the eigenvalue to be used
           e1 -- first estimate of binding energy (should be negative)
           e2 -- second estimate of binding energy (should be negative)
           elow -- largest energy to be used in search (should be negative)
           tol -- if two consecutive energies differ by less then tol, the search is converged
           
           Energies are given in fm**-1. """
        
        # determine eigenvalues for starting energies        
        eta1,pgrid,pweight,wf=self.eigv(e1,neigv)
        eta2,pgrid,pweight,wf=self.eigv(e2,neigv)
        
        while abs(e1-e2) > tol: 
          # get new estimate (taking upper value into account)   
          enew=e2+(e1-e2)/(eta1-eta2)*(1-eta2) 
          enew=min(elow,enew)
       
          # get new eigenvalue and replace e1 and e2 for next iteration
          eta,pgrid,pweight,wf=self.eigv(enew,neigv)
          e2=e1
          eta2=eta1
          e1=enew
          eta1=eta 
            
        return e1,eta1,pgrid,pweight, wf 
           
    def fourier(self,wfp):
        """Calculates the Fourier transform of the partial wave representation of the wave function.
        
           Parameter: 
           wfp -- wave function in momentum space
            
           Note that the factor I**l is omitted."""
        
        # calculate spherical bessel functions based dense Fourier trafo momentum grid and rgrid
        # prepare matrix based on r,p points  
        rpmat = np.outer(self.rgrid,self.pfourgrid)
        # evaluate jl     
        jlmat = scipy.special.spherical_jn(self.l,rpmat)
        
        # interpolate of wave to denser Fourier trafo grid
        wfinter = interpolate.interp1d(self.pgrid, wfp, kind='cubic',fill_value="extrapolate")
        # interpolate wf and multiply my p**2*w elementwise 
        wfdense = wfinter(self.pfourgrid)*self.pfourgrid**2*self.pfourweight*np.sqrt(2/m.pi)
        
        # now the Fourier trafo is a matrix-vector multiplication 
        wfr = jlmat.dot(wfdense)
        
        return self.rgrid,wfr
    
    
    def rms(self,wfr):
        """Calculates the norm and rms radius for the given r-space wave function.
        
           Normalization of the wave function is assumed. 
           Parameter: 
           wfr -- wave function in r-space obtained by previous Fourier trafo"""
        
        
        norm=np.sum(wfr**2*self.rweight*self.rgrid**2)
        rms=np.sum(wfr**2*self.rweight*self.rgrid**4)

            
        rms=np.sqrt(rms)
        
        return norm,rms


def formfactor(q, pgrid, pweight, psi, nang, l, lz):
    ''' Computes Formfactor of deuterion as a function of the cutoff Lambda and energy-momentum of the 
    exchanged photon q^2
    input:
    q         absolute value of vector q (in z direction) in fm**-1
    pgrid     grid points for p
    pweight   weights of pgrid
    psi       wavefunction for certain Lambda
    nang      number of grid points for angular momentum integration
    l, lz     angular momentum (z comp)
    '''


    psistar = np.conj(psi)

    tck = interpolate.splrep(pgrid, psi, s=0)

    #find correct spherical harmonics
    def sph_harm_q(x):
      phi = 0
      theta = np.arccos((pgrid*x - 1./2.*q)/np.sqrt(pgrid**2*(1-x**2) + (pgrid*x - 1./2.*q)**2)) 
      return scipy.special.sph_harm(lz, l, phi, theta) #l = lz = 0

    def integral(x):
      pgrid_new = np.sqrt((pgrid*np.sqrt(1-x**2))**2 + (pgrid*x - 0.5*q)**2)
      psinew = interpolate.splev(pgrid_new, tck, der=0)
      return np.real(psinew*psistar*sph_harm_q(x)*scipy.special.sph_harm(lz, l, 0, np.arccos(x)))

    #Now integrate over x
    #print(psinew)

    xgrid,xweight=leggauss(nang)

    integ = 0
    for i, x in enumerate(xgrid):
      integ += xweight[i] * integral(x)

    #Now integrate over p prime
    F = np.sum(pweight*pgrid**2*integ)

    return 2*np.pi*F


Lamlist = [300.0,400.0, 500.0,600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]
C0list = [-9.827953e-02, -2.820315e-02, -4.221894e-04, 1.285743e-02, 2.016719e-02, 2.470795e-02, 2.786520e-02, 3.030801e-02, 3.239034e-02, 3.431611e-02]
l = 0
lz = 0

psilist = []
pgridlist = []
pweightlist = []

#extract pgrid, pweight and wave functions from data given
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=300.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=400.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=500.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=600.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=700.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=800.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=900.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=1000.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=1100.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)
pgrid, pweight, psi = np.loadtxt("data/wf-obe-lam=1200.00.dat", skiprows = 2).transpose()
psilist.append(psi)
pgridlist.append(pgrid)
pweightlist.append(pweight)



###4. check numerical accuracy of result


#start with high q
nangle = np.arange(10, 100, 10)
qvec = np.arange(0, 10, 1)
error = 1e-6

for q in qvec:
  prev  = 0
  print("q = ", q)
  for nang in nangle:
    new = formfactor(q, pgridlist[-1], pweightlist[-1], psilist[-1], nang, l, lz)
    if abs(new-prev) > error:
      print("not stable for q = %s, nang = %s"%(q, nang))
      prev = new

# --> numerically stable even for higher q (stability up to 1e-6). nang = 90 suffices

nang = 90

###5. check for normalization and derivative

#calculated results
F0 = formfactor(0., pgridlist[-1], pweightlist[-1], psilist[-1], nang, l, lz)
print("F(0) = %s"%F0)

pot=OBEpot(nx=24,mpi=138.0,C0=C0list[-1],A=-1.0/6.474860194946856,cutoff=Lamlist[-1])
solver=TwoBody(pot=pot,np1=40,np2=20,pa=1.0,pb=7.0,pc=35.0,mred=938.92/2,l=0,
                            nr1=40, nr2=20, ra=1.0, rb=5.0, rc=20.0, 
                            np1four=400,np2four=200)

rgrid, wfr = solver.fourier(psilist[-1])
r2 = solver.rms(wfr)

print("<r^2> = ", r2)


#find radius squared from fit to data points
def quadr(x, a):
  return 1. - 1./6.*a*x**2 

qray = np.linspace(0, 1, 20)
F = np.zeros(len(qray))

for i,q in enumerate(qray):
  F[i] = formfactor(q, pgridlist[-1], pweightlist[-1], psilist[-1], nang, l, lz)

#perform fit using curve_fit function from scipy optimize
par, error = curve_fit(quadr, qray, F)

print(par[0], np.sqrt(error[0][0]))

x = np.linspace(0, 1, 100)

fig_radius = plt.figure()
ax = plt.gca()
  
ax.plot(qray, F,
        label = (r"calculated"),
        marker = "o",
        linestyle = "none")
ax.plot(x, quadr(x, *par), label= "fit", linestyle = "-")

ax.set_xlabel(r"momentum $q$")
ax.set_ylabel(r"Formfactor $F(\vec{q}^2)$")
ax.legend(loc=0)
ax.grid(True)
fig_radius.tight_layout()

fig_radius.savefig("formfactor_radius.pdf")


###6. plot formfactors for several lambda 
qray = np.linspace(0, 10, 50)

fig_lambda = plt.figure()
ax = plt.gca()

for i in range(len(Lamlist)):
    F = np.zeros(len(qray))
    for j, q in enumerate(qray):
      F[j] = formfactor(q, pgridlist[i], pweightlist[i], psilist[i], nang, l, lz)
    ax.plot(qray**2, F,
            label = (r"$\Lambda = %s$"%Lamlist[i]),
            linestyle = "-")

ax.set_xlabel(r"momentum $q^2$")
ax.set_ylabel(r"Formfactor $F(\vec{q}^2)$")
ax.legend(loc=0)
ax.grid(True)
fig_lambda.tight_layout()

fig_lambda.savefig("formfactor_lambda.pdf")
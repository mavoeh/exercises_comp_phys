from ThreeBodyScattering import OBEpot, TwoBody
from fit import fitc0
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import ExpressionModel

class TwoBodyScatt(TwoBody):
    
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, mred=938.92/2,l=0,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            np1four=200,np2four=100):
        
        
        # first use the TwoBody class to keep the main parameters 
        super().__init__(pot,np1,np2,pa,pb,pc,mred,l,nr1,nr2,ra,rb,rc,np1four,np2four)

        # append one momentum to grid for the onshell value
        # value will be defined during solution in lseq below
        self.pgrid=np.append(self.pgrid,[0.0])
        self.pweight=np.append(self.pweight,[0.0])
        
# now turn to scattering and solve for LS equation to get tmatrix (on- and offshell)
    def lseq(self,E):
      # set up equations for bound state
    
      # on-shell momentum  
      pon=np.sqrt(2*self.mred*E)
      self.pgrid[self.npoints]=pon
        
      # define matrix for set of equations 
      # predefine the Kronecker deltas 
      amat=np.identity(self.npoints+1,dtype=np.cdouble)
      # now add the other pieces of the definition of the matrix   
      for i in range(self.npoints+1):
        # first for j != N 
        for j in range(self.npoints):   
          amat[i,j]+=-(2*self.mred)*self.pot.v(self.pgrid[i],self.pgrid[j],self.l)*self.pgrid[j]**2 \
                               /(pon**2-self.pgrid[j]**2)*self.pweight[j]  \
        # then for j==N              
        amat[i,self.npoints]  \
           +=(2*self.mred)*self.pot.v(self.pgrid[i],pon,self.l)*pon**2* \
                 np.sum(self.pweight[0:self.npoints-1]/(pon**2-self.pgrid[0:self.npoints-1]**2))  \
             +1j*np.pi*self.mred*pon*self.pot.v(self.pgrid[i],pon,self.l)  \
             -self.mred*pon*self.pot.v(self.pgrid[i],pon,self.l)*np.log(abs((pon+self.pc)/(self.pc-pon)))
        
      # now define the rhs   
      bmat=np.empty((self.npoints+1,self.npoints+1),dtype=np.cdouble)
      for i in range(self.npoints+1):
        for j in range(self.npoints+1):   
            bmat[i,j]=self.pot.v(self.pgrid[i],self.pgrid[j],self.l)
            
      # finally solve set of equations 
      tmat=np.linalg.solve(amat,bmat)
        
      # return onshell matrix element  
      return tmat[self.npoints,self.npoints]
    
    def wq(self,tonshell,theta):
      """Calculates differential cross section based on list of tmatrix elements. 
      
         Parameter: 
         tonshell -- list of complex pw matrix elements of the t-matrix. Maximum l is len(tonshell)-1.
         theta -- array of angles in rad for which the wq needs to be calculated.
         
         Returns array of the cross section values for the different theta. 
         
      """
   
      tmatsum=np.zeros((len(theta)),dtype=np.cdouble)
      xval=np.cos(theta)
        
      for l in range(len(tonshell)):
        leg=legendre(l)
        tmatsum+=(2*l+1)*leg(xval)*tonshell[l]
        
        
      return m.pi**2*self.mred**2*np.abs(tmatsum)**2    

# set number of energies used for extraction of a,r 
numk=10

# prepare parameters 
# get figure enviroment 
fig, ax = plt.subplots()

# set labels 
ax.set_xlabel(r'$E_{CM}$[MeV]')
ax.set_ylabel(r'$\delta(E)$[deg]')

# set some limits on axes to concentrate on most relevant part of the interaction. 

#ax.set_xlim(0,100)
#ax.set_ylim(1E-6,15)

energies=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,4,10,20,30,50,100,200,300,500]

Lam = [700.0]
e0list = [-2.125, -2.225, -2.325]

a = []
r = []


for e0 in e0list:
    # determine interacion and set up solver 
    para = fitc0(Lam,e0)[0]
    pot=OBEpot(nx=24,mpi=138.0,C0=para[1],A=-1.0/6.474860194946856,cutoff=para[0])
    fbsolver=TwoBodyScatt(pot=pot,np1=40,np2=20,pa=1.0,pb=7.0,pc=35.0,mred=938.92/2,l=0,
                            nr1=40, nr2=20, ra=1.0, rb=5.0, rc=20.0, 
                            np1four=400,np2four=200)
    
    plt_delta=[]
    for ener in energies:
      # perform energy search for this parameter set
      tonshell=fbsolver.lseq(ener/TwoBody.hbarc)
      smat=1-1j*938.92/197.327*np.pi*np.sqrt(ener*938.92/TwoBody.hbarc**2)*tonshell
      delta=np.real(-0.5*1j*np.log(smat))*180.0/np.pi
      if ener < 10 and delta < 0:
         delta+=180.0
    
      plt_delta.append(delta)
        
      # print phase shift 
      # print("{3:15.3f} {0:15.6e}   {1:15.6e}  {2:15.6e}".format(ener,delta,np.abs(smat),para[0]))

    # phase shifts for one energie are collected 
    # plot these phases 
    ax.plot(energies,plt_delta,label=r"$\Lambda=$ {0:10.3f}".format(para[0]))

    
    # now use the first n phases to determine a,r
    # calc momenta from CM energies
    deltarad=np.array(plt_delta)*np.pi/180.0
    eval=np.array(energies)/TwoBody.hbarc
    k=np.sqrt(938.92/TwoBody.hbarc*eval)
    kcotdelta=k/np.tan(deltarad)

    mod = ExpressionModel('-1/a+0.5*r*x**2+C*x**4')
    res = mod.fit(kcotdelta[0:numk], a=5,C=0,r=1, x=k[0:numk])
    print(res.fit_report())
    
    a.append(res.params["a"].value)
    r.append(res.params["r"].value)
    
    
ax.legend(loc="best")
    
#fig.savefig("phases.pdf")   

print(a)
print(r) 

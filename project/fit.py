import numpy as np
from ThreeBodyScattering import *

def fitc0(lamlist, e0, nx=16,np1=20,np2=10,pa=1.0, pb=5.0, pc=20.0):
    """Fits C0 values for given set of discretizations.
    
       Parameters:
       lamlist -- list of Lambda values in MeV
       nx -- number of x grid points for partial wave projection of the potential
       np1,np2 -- grid points for momentum grid for the solution of the Schrödinger equation
       pa,pb,pc -- limits of intervals for the momentum grid 
    """  
    
    # fix start value for C0 and parameters of long range part etc. 
    c1=0.0
    A=-1.0/6.474860194946856
    mpi=138.0
    mred=938.92/2
    l=0
    e0=e0/TwoBody.hbarc
    neigv=1
    
    # fit results will be collected in fitres
    fitres=[]
    for Lambda in lamlist: 
      # c1 is reused from previous lambda  
      c2=c1+0.1 
    
      # set up potential 
      pot=OBEpot(nx=nx,mpi=mpi,C0=c1,A=A,cutoff=Lambda)
      #set up solver for two-body problem
      solver=TwoBody(pot=pot,np1=np1,np2=np2,pa=pa,pb=pb,pc=pc,mred=mred,l=l)  
      # determine eigenvalue for the given energy 
      eta1,pgrid,wf=solver.eigv(e0,neigv)

      # same as above for second value of C0 
      pot=OBEpot(nx=nx,mpi=mpi,C0=c2,A=A,cutoff=Lambda)
      solver=TwoBody(pot=pot,np1=np1,np2=np2,pa=pa,pb=pb,pc=pc,mred=mred,l=l)    
      eta2,pgrid,wf=solver.eigv(e0,neigv)
        
      # repeat the steps of the second method until there is no change in c anymore   
      while abs(c1-c2) > 1E-10: 
          # get new estimate (taking upper value into account)   
          cnew=c2+(c1-c2)/(eta1-eta2)*(1-eta2) 

          # repeat the determination of the eigenvalue for new c 
          pot=OBEpot(nx=nx,mpi=mpi,C0=cnew,A=A,cutoff=Lambda)
          solver=TwoBody(pot=pot,np1=np1,np2=np2,pa=pa,pb=pb,pc=pc,mred=mred,l=l)    
          eta,pgrid,wf=solver.eigv(e0,neigv)
        
          # shift c1,c2,cnew for the secant method's iterations
          c2=c1
          eta2=eta1
          c1=cnew
          eta1=eta 
            
      # collect Lambda and corresponding values of C0       
      fitres.append([Lambda,cnew])
      # print the result (to see some progress during fit)
      print("{0:15.3f}  {1:15.6e}  {2:15.6e}".format(Lambda,cnew,eta))
        
    return fitres 

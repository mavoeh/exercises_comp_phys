import numpy as np
import matplotlib.pyplot as plt

# prepare interpolation using cubic hermitian splines 

class Cubherm:
    """Prepares spline functions for cubic hermitian splines. 
    
    see Hueber et al. FBS 22,107 (1997). 
    
    The function spl returns the the spline function for a given x. 
    If x is below the smallest grid point, extrapolation is used. 
    If x is after largest grid point, then the function evaluates to zero. 
    """
    
        
    def spl(xold,xin):
        """Calculates spline functions for given values xold and xnew.
        
           Parameters:
           xold -- set of grid points where function is given. xold needs to be one dimensional.
           xin  -- set of grid points to interpolate to. xnew can be multidimensional. 
           
           On return spline functions will be given that have the shape of xnew and xold combined. 
        """
        
        # first determine the base value of the index for each xnew.
        
        nold=len(xold)
        if nold<4:
          raise(ValueError("Interpolation requires at least 4 grid points.")) 
        
        xnew=xin.reshape((-1))        
        indx=np.empty((len(xnew)),dtype=np.int)
        
        for i in range(len(xnew)):
          # do not extrapolated beyond largest grid point
          if xnew[i] > xold[nold-1]: 
            indx[i]=-1
          else:  
            for j in range(nold):
              if xnew[i] <= xold[j]:
                break          
            if j < 1:
              indx[i]=0
            elif j > nold-3:
              indx[i]=nold-3
            else:
              indx[i]=j-1  

        # then prepare phi polynomials for each x 
        
        phi1=np.zeros((len(xnew)),dtype=np.double)
        phi2=np.zeros((len(xnew)),dtype=np.double)
        phi3=np.zeros((len(xnew)),dtype=np.double)
        phi4=np.zeros((len(xnew)),dtype=np.double)
        
        for i in range(len(xnew)):
          if indx[i]>0:  
            phi1[i] = (xold[indx[i] + 1] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 3 * (xold[indx[i] + 1] - 3 * xold[indx[i]] + 2 * xnew[i])
            phi2[i] = (xold[indx[i]] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 3 * (3 * xold[indx[i] + 1] - xold[indx[i]] - 2 * xnew[i])
            phi3[i] = (xnew[i] - xold[indx[i]]) * (xold[indx[i] + 1] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 2
            phi4[i] = (xnew[i] - xold[indx[i] + 1]) * (xold[indx[i]] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 2
        
        # now we are ready to prepare the spline functions 
        # most are zero 
        splfu=np.zeros((len(xold),len(xnew)),dtype=np.double)
        for i in range(len(xnew)):
          if indx[i]>0:  
            splfu[indx[i]-1,i] = \
               -phi3[i]*(xold[indx[i]+1]-xold[indx[i]])/(
                        (xold[indx[i]]-xold[indx[i]-1])*(xold[indx[i]+1]-xold[indx[i]-1]))
            
            splfu[indx[i],i] = phi1[i] \
                +phi3[i]*((xold[indx[i]+1]-xold[indx[i]])/ (xold[indx[i]]-xold[indx[i]-1]) \
                         -(xold[indx[i]]-xold[indx[i]-1])/ (xold[indx[i]+1]-xold[indx[i]]))/(xold[indx[i]+1]-xold[indx[i]-1]) \
                -phi4[i]*(xold[indx[i]+2]-xold[indx[i]+1])/ (xold[indx[i]+1]-xold[indx[i]])/(xold[indx[i]+2]-xold[indx[i]])

            splfu[indx[i]+1,i] = phi2[i] \
                +phi3[i]*(xold[indx[i]]-xold[indx[i]-1])/ (xold[indx[i]+1]-xold[indx[i]])/(xold[indx[i]+1]-xold[indx[i]-1]) \
                +phi4[i]*((xold[indx[i]+2]-xold[indx[i]+1])/ (xold[indx[i]+1]-xold[indx[i]]) \
                         -(xold[indx[i]+1]-xold[indx[i]])/ (xold[indx[i]+2]-xold[indx[i]+1]))/(xold[indx[i]+2]-xold[indx[i]])
            
            
            splfu[indx[i]+2,i] = \
                phi4[i]*(xold[indx[i]+1]-xold[indx[i]])/(
                        (xold[indx[i]+2]-xold[indx[i]+1])*(xold[indx[i]+2]-xold[indx[i]]))
          elif indx[i]>=0:
            # do linear interpolation at the origin 
            splfu[indx[i],i] = (xnew[i]-xold[indx[i]+1])/(xold[indx[i]]-xold[indx[i]+1]) 
            splfu[indx[i]+1,i] = (xold[indx[i]]-xnew[i])/(xold[indx[i]]-xold[indx[i]+1]) 

        retshape=[nold] 
        for n in list(np.shape(xin)):
          retshape.append(n)
        
        return splfu.reshape(retshape)

def bilin_int1(func, xg, yg, x, y):
  '''find grid points closest to x and y, so that we can evaluate at the right grid point
  and interpolate array if necessary'''

  #find closest grid points
  ix = np.abs(xg - x).argmin()
  iy = np.abs(yg - y).argmin()

  if(xg[ix] > x or ix == len(xg)-1):
  	print(xg[ix])
  	ix -= 1
  if(yg[iy] > y or iy == len(yg)-1):
  	print(yg[iy])
  	iy -= 1

  ixm = ix+1
  iym = iy+1
  
  x1 = xg[ix]
  y1 = yg[iy]
  x2 = xg[ixm]
  y2 = yg[iym]
  f11 = func[ix, iy]
  f12 = func[ix, iym]
  f21 = func[ixm, iy]
  f22 = func[ixm, iym]

  fxy = 1./(x2-x1)/(y2-y1)*(f11*(x2-x)*(y2-y) + f21*(x-x1)*(y2-y) + f12*(x2-x)*(y-y1) + f22*(x-x1)*(y-y1))

  return fxy

def interpol(func, xg, yg, x, y):
  '''find grid points closest to x and y, so that we can evaluate at the right grid point
  and interpolate array if necessary'''

  #find index closest to x and y
  ix = np.abs(xg - x).argmin()
  iy = np.abs(yg - y).argmin()

  #define new grid 
  xgnew = xg + (xg[ix]-x)
  ygnew = yg +(yg[iy] -y)

  #interpolate function
  xspl = Cubherm.spl(xg, xgnew)
  yspl = Cubherm.spl(yg, ygnew)

  int1 = np.empty((len(xg), len(yg)), dtype = np.cdouble)
  int2 = np.empty((len(xg), len(yg)), dtype = np.cdouble)
  for i in range(len(xg)):
    for j in range(len(yg)):
      int1[i,j] = np.sum(func[0:len(xg), j]*xspl[0:len(xg), i])

  for i in range(len(xg)):
    for j in range(len(yg)):
      int2[i,j] = np.sum(int1[i, 0:len(yg)]*yspl[0:len(yg), j])

  return int2[ix, iy]

#try bilin_int1

xc = np.linspace(0,100,11)
xf = np.linspace(0,100,101)

yc = np.linspace(0,100,11)
yf = np.linspace(0,100,101)

fc = np.empty((len(xc), len(yc)), dtype = np.float)
for i in range(len(xc)):
	for j in range(len(yc)):
		fc[i,j] = xc[i]**2 + yc[j]**2

ff = np.empty((len(xf), len(yf)), dtype = np.float)
for i in range(len(xf)):
	for j in range(len(yf)):
		ff[i,j] = xf[i]**2 + yf[j]**2


print(interpol(fc, xc, yc, 24, 99))
print(ff[24, 99])
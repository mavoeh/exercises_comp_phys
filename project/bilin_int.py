import numpy as np
import matplotlib.pyplot as plt

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

#try bilin_int1

xc = np.linspace(0,100,101)
xf = np.linspace(0,100,1001)

yc = np.linspace(0,100,101)
yf = np.linspace(0,100,1001)

fc = np.empty((len(xc), len(yc)), dtype = np.float)
for i in range(len(xc)):
	for j in range(len(yc)):
		fc[i,j] = xc[i]**2 + yc[j]**2

ff = np.empty((len(xf), len(yf)), dtype = np.float)
for i in range(len(xf)):
	for j in range(len(yf)):
		ff[i,j] = xf[i]**2 + yf[j]**2


print(bilin_int1(fc, xc, yc, 24, 99))
print(ff[240, 990])
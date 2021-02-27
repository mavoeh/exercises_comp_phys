import numpy as np
import matplotlib.pyplot as plt

def scurve(theta1, theta2, phi, Elab, m = 938.92, e = -2.225, N = 10**5+1, deg = False):
    """
    Calculates the S curve for a given set of scattering
    angles and energy of the incoming particle
    
    theta1  -   Scattering angle of particle 1 relative to z-axis.
    theta2  -   Scattering angle of particle 2 relative to z-axis.
    phi     -   Difference between scattering angles of
                Particles 1 and 2 in the x-y-plane.
    Elab    -   Energy of the incoming particle.
    m       -   Mass of the particles.
    e       -   Binding energy of the two-body bound state.
    N       -   Number of points for discretization of the ellipse.
                (Note that if the ellipse is partially negative in
                k1 or k2, due to removal of the unphysical values
                the returned S curve contains less points.)
                
    returns: S, k1, k2
    
    S   -   Arclength of the ellipse in energy space.
            Starting point defined as in [1] (p.127 and App.B)      <-- starting point not fully implemented yet!
    k1  -   Absolute value of the momentum of particle 1 at the
            corresponding value of S.
    k2  -   Absolute value of the momentum of particle 2 at the
            corresponding value of S.
    ___________________________________________________________________
    
    [1] -   W. Glöckle, H. Witala, D. Hüber, H. Kamada, and J. Golak,
            "The Three nucleon continuum: Achievements, challenges
             and applications" Phys. Rept. 274 (1996) 107–285.
    """
    # calculate momentum of incoming particle from its energy
    k0 = np.sqrt(2*938.92*Elab)
    
    # transform input angles to radian if deg is True
    if deg == True:
        theta1 *= np.pi/180
        theta2 *= np.pi/180
        phi *= np.pi/180
    
    # check if angles between 0 and pi (theta) or 0 and 2pi (phi)
    if (theta1<0)or(theta1>np.pi)or(theta2<0)or(theta2>np.pi)or(phi<0)or(phi>2*np.pi):
        raise ValueError("Angles should be between 0 and pi!")
    
    # define theta_m and theta_p to distinguish between different cases
    theta_m = np.arccos(np.sqrt(4*m*np.abs(e)/k0**2))
    theta_p = -np.arccos(np.sqrt(4*m*np.abs(e)/k0**2))+np.pi
    """
    ###########test
    theta1 = theta_m+0.01
    theta2 = theta_m+0.01
    """
    # check if the set of input angles is mathematically allowed
    if (theta1-np.pi/2)**2 + (theta2-np.pi/2)**2 < (theta_p-np.pi/2)**2:
        raise ValueError("Configuration of scattering angles mathematically forbidden!")
    
    # check if input angles yield positive momenta
    if ((theta1>np.pi/2 and theta2>theta_m) or (theta1>theta_m and theta2>np.pi/2)):
        return [[]]*3
    
    # analytically calculate semi-axis and displacement    
    c1, s1, c2, s2 = np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2)
    z = np.cos(phi)*s1*s2+c1*c2
    a = np.sqrt( (2*m*e*(4-z**2)+2*k0**2*(c1**2+c2**2-z*c1*c2)) / (z**3-2*z**2-4*z+8) )
    b = np.sqrt( (2*m*e*(z**2-4)-2*k0**2*(c1**2+c2**2-z*c1*c2)) / (z**3+2*z**2-4*z-8) )
    x0 = k0*(z*c2-2*c1)/(z**2-4)
    y0 = k0*(z*c1-2*c2)/(z**2-4)
    
    # starting angle for parametrization at an angle of -3/4pi
    # (ellipse is already rotated by -pi/4)
    t0 = -np.pi/2
    
    # now calculate parametrization
    t = np.linspace(t0, t0+2*np.pi, N)
    k = np.zeros((N,2))
    k[:,0] = 1/np.sqrt(2) * ( a*np.cos(t) + b*np.sin(t)) + x0
    k[:,1] = 1/np.sqrt(2) * (-a*np.cos(t) + b*np.sin(t)) + y0
    
    # select only physically relevant values
    mask = (k[:,0] >= 0) & (k[:,1] >= 0)
    k1 = k[:,0][mask]
    k2 = k[:,1][mask]
    
    indices = np.array(np.where(mask)[0]) # array of indices where kx and ky both positive
    #print(len(indices)==max(indices)-min(indices)+1)
    
    # calculate the arclength corresponding to the point (k1,k2)
    S = np.zeros(k1.shape)
    S[1:] = np.cumsum( np.sqrt( k1[1:]**2*(k1[1:]-k1[:-1])**2 + k2[1:]**2*(k2[1:]-k2[:-1])**2 )/m \
    * np.floor(1/(indices[1:]-indices[:-1])) # <- this line = 0, if there is a discontinuity in
    )                                        # k1 or k2, due to removal of unpysical values,
                                             # so that S does not get increased; = 1 else
    return S, k1, k2
    

#S, kx, ky = scurve(35,0,90,1,1,-0.18,deg=True)

S, kx, ky = scurve(41.9,41.9,180,22.7,deg=True)
print(S[-1])

S, kx, ky = scurve(44,44,180,65,deg=True)
print(S[-1])

S, kx, ky = scurve(20,116.2,180,65,deg=True)
print(S[-1])

plt.plot(kx, ky)
plt.gca().axis('equal')
plt.scatter(kx[0],ky[0])
n = np.linspace(0,max(kx),len(S))
plt.plot(n, S)

plt.show()

"""
n = np.linspace(10, 80)
n = np.round(1.2**n).astype(np.int)
print(n)
Slist = []
for N in n:
    print(N)
    S, kx, ky = scurve(15,90,90,k0=1,e=1,m=1,N=N)
    Slist.append(S[-1])

plt.scatter(n, Slist)
plt.grid(True)
plt.title("Convergence of S-curve")
plt.xlabel("N", fontsize=14)
plt.ylabel("Total arclength",fontsize=14)
plt.loglog()
plt.show()
"""

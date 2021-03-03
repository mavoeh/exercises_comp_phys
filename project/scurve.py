import numpy as np

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
    deg     -   Boolean value: if True, the input angles should be
                given in degree, if False angles should be in radian.
                
    returns: S, k1, k2, t
    
    S   -   numpy.array: Arclength of the ellipse in energy space.
            Starting point defined as in [1] (p.127 and App.B)
            Note: If the complete ellipse is in the first quadrant,
            the starting point is defined as the point, where a line
            with a slope of 1, passing through the center of the
            ellipse, would cut the ellipse.
    k1  -   numpy.array: Absolute value of the momentum of
            particle 1 at the corresponding value of S.
    k2  -   numpy.array: Absolute value of the momentum of
            particle 2 at the corresponding value of S.
    t   -   Value of the parameter used to parametrize the ellipse,
            at the corrseponding value of S.
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
    
    # starting angle for parametrization at an angle of 0
    # (ellipse is already rotated by -pi/4)
    t0 = np.pi/4
    # except if whole ellipse is in first quadrant: start at -3pi/4
    if (theta1 >= theta_m) and (theta2 >= theta_m):
        t0 = -np.pi/2
        
    k10 = 1/np.sqrt(2) * ( a*np.cos(t0) + b*np.sin(t0)) + x0
    k20 = 1/np.sqrt(2) * (-a*np.cos(t0) + b*np.sin(t0)) + y0
    
    # now calculate parametrization
    t = np.linspace(t0, t0+2*np.pi, N)
    k = np.zeros((N,2))
    k[:,0] = 1/np.sqrt(2) * ( a*np.cos(t) + b*np.sin(t)) + x0
    k[:,1] = 1/np.sqrt(2) * (-a*np.cos(t) + b*np.sin(t)) + y0
    
    # select only physically relevant values
    mask = (k[:,0] >= 0) & (k[:,1] >= 0)
    k1 = k[:,0][mask]
    k2 = k[:,1][mask]
    t = t[mask]
    
    # indices where kx and ky both positive
    indices = np.array(np.where(mask)[0])
    
    # define d[i] = number of skipped gridpoints in t,
    # to get from (k1[i], k2[i]) to (k1[(i+1)%N], k2[(i+1)%N])
    d = np.zeros(k1.shape)
    if indices[-1] != N-1:
        d[-1] = N - indices[-1] + indices[0] - 1
    d[:-1] = (indices[1:]-indices[:-1]) - 1
    
    # if there is a discontinuity, choose S = 0 after the last discontinuity,
    # by shifting the numpy arrays by the corresponding index
    if np.any( d != np.zeros(len(d)) ):
        i0 = np.argwhere(d > 0)[-1] + 1
        k1 = np.roll(k1, -i0)
        k2 = np.roll(k2, -i0)
        t  = np.roll(t,  -i0)
        d  = np.roll(d,  -i0)
    
    # calculate the arclength corresponding to the point (k1,k2)
    S = np.zeros(k1.shape)
    S[1:] = np.cumsum( np.sqrt( k1[1:]**2*(k1[1:]-k1[:-1])**2 + k2[1:]**2*(k2[1:]-k2[:-1])**2 )/m \
    * np.floor(1/(d[:-1] + 1))      # <------- this line = 0, if there is a discontinuity in
    )                                        # k1 or k2, due to removal of unpysical values,
                                             # so that S does not get increased; = 1 else
    return S, k1, k2, t

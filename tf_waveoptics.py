# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:30:30 2018

@author: Deniz Mengu, UCLA
"""

import tensorflow as tf
import numpy as np

def tf_flattop2(f0,dfx,dfy,fxy2):
    
    M = tf.to_float(tf.shape(fxy2)[0])
    N = tf.to_float(tf.shape(fxy2)[1])
    drho = tf.sqrt(dfx**2+dfy**2)
    rho = tf.sqrt(fxy2)

    Q = tf.to_float((fxy2)<=(f0**2))
    alpha = M*N/tf.reduce_sum(Q)
    alpha = tf.reduce_min([M,N])/2/tf.sqrt(alpha)/10.0
    T = tf.exp((-(rho-f0)**2)/2.0/((alpha*drho)**2))
    W = Q+T
    flattop_filter = tf.clip_by_value(W,clip_value_min=0.0,clip_value_max=1.0)
    
    return flattop_filter

def adaptive_bandlimit(fx,fy,z,dfx,dfy,wlength,theta_max):

    # fx,fy -> frequency grid
    # z -> propagation distance
    # Sx, Sy -> width of signal region at the input plane
    # Wx, Wy -> width of the output plane window
    
    FX = tf.matmul(fx,tf.ones_like(fy))
    FY = tf.matmul(tf.ones_like(fx),fy) 
    fx_lim = tf.divide(1/wlength,(((2*dfx*z)**2+1)**0.5))
    fy_lim = tf.divide(1/wlength,(((2*dfy*z)**2+1)**0.5))

    theta_max = theta_max/180.0*np.pi # deg2rad
    fx_lim_t = tf.cast(tf.sin(theta_max)/wlength,tf.float32)
    fy_lim_t = tf.cast(tf.sin(theta_max)/wlength,tf.float32)
    fR = tf.square(fx_lim_t)+tf.square(fy_lim_t)
    Qt = tf.to_float(tf.less((tf.square(FX)+tf.square(FY)),(fR)))

    Q = tf.multiply(tf.to_float(tf.less_equal(tf.square(FX)/fx_lim**2+tf.square(FY)*wlength**2,1.0)),\
        tf.to_float(tf.less_equal(tf.square(FY)/fy_lim**2+tf.square(FX)*wlength**2,1.0)))

    Q = tf.multiply(Q,Qt)

    return Q

def paraxlens(wlength,focal_length,center,fshift,diameter,dx,dy,M,N):
    
    """
    Create function of a ideal paraxial lens
    INPUTS : 
        wlength -> wavelength of the optical signal
        f -> 2-by-1 list contains -> focal length in x and focal length in y
        center -> 2-by-1 list contains -> central shift of whole lens func. in x and y
        fshift -> lateral off-axis shift on the focal plane
        dia -> diameter of lens aperture
        dx,dy -> sampling intervals in space
        M,N -> size of simulation window
        
    OUTPUTS:
        lensfunc -> numerical lensfunction defined in space
    """
    radius = diameter/2
    x = tf.expand_dims((tf.range(M,dtype=tf.float32)-(M-1)/2)*dx,-1)
    y = tf.expand_dims((tf.range(N,dtype=tf.float32)-(N-1)/2)*dy,0)
    x0,y0 = center[0], center[1]
    xf,yf = fshift[0], fshift[1]
    Lxy2 = tf.matmul((x-x0)**2,tf.ones((1,N)))+tf.matmul(tf.ones((M,1)),(y-y0)**2)
    D = tf.to_float(tf.less_equal(Lxy2,radius**2))
    sq = tf.matmul((x-xf)**2,tf.ones((1,N)))+tf.matmul(tf.ones((M,1)),(y-yf)**2)
    lens_phase = -2*np.pi/wlength*sq/2/focal_length
    lensfunc = tf.complex(D*tf.cos(lens_phase),D*tf.sin(lens_phase))
    return lensfunc

def ideallens(wlength,focal_length,center,fshift,diameter,dx,dy,M,N):
    
    """
    Create function of a ideal paraxial lens
    INPUTS : 
        wlength -> wavelength of the optical signal
        f -> 2-by-1 list contains -> focal length in x and focal length in y
        center -> 2-by-1 list contains -> central shift of whole lens func. in x and y
        fshift -> lateral off-axis shift on the focal plane
        dia -> diameter of lens aperture
        dx,dy -> sampling intervals in space
        M,N -> size of simulation window
        
    OUTPUTS:
        lensfunc -> numerical lensfunction defined in space
        
    """
    radius = diameter/2
    x = tf.expand_dims((tf.range(M,dtype=tf.float32)-(M-1)/2)*dx,-1)
    y = tf.expand_dims((tf.range(N,dtype=tf.float32)-(N-1)/2)*dy,0)
    x0,y0 = center[0], center[1]
    xf,yf = fshift[0], fshift[1]
    Lxy2 = tf.matmul((x-x0)**2,tf.ones((1,N)))+tf.matmul(tf.ones((M,1)),(y-y0)**2)
    D = tf.to_float(tf.less_equal(Lxy2,radius**2))
    sq = tf.matmul((x-xf)**2,tf.ones((1,N)))+tf.matmul(tf.ones((M,1)),(y-yf)**2)
    R= tf.sqrt(sq+focal_length**2)
    lens_phase = -2*np.pi/wlength*R
    lensfunc = tf.complex(D*tf.cos(lens_phase),D*tf.sin(lens_phase))
    return lensfunc

class NHW_FSPAS_FFT:

    def __init__(self, field, wlength, z, dx, dy, ridx, theta_max):

        self.B, self.M, self.N = tf.shape(field)[0],tf.to_float(tf.shape(field)[1]),tf.to_float(tf.shape(field)[2])
        self.output, self.scattered_pwr = self.tf_FSPAS_FFT(field, wlength, z, dx, dy, ridx, theta_max)

    def tf_fft_shift_2d(self, field):

        B, M, N = self.B, self.M, self.N
        def fM0(): return M/2
        def fM1(): return tf.floor(M/2)+1
        M_half = tf.to_int32(tf.cond(tf.equal(tf.mod(M, 2),0), fM0, fM1))
        def fN0(): return N/2
        def fN1(): return tf.floor(N/2)+1
        N_half = tf.to_int32(tf.cond(tf.equal(tf.mod(N, 2),0), fN0, fN1))

        N = tf.to_int32(N)
        M = tf.to_int32(M)
        img_UL = tf.slice(field, tf.stack([0, 0, 0]), tf.stack([B, M_half, N_half]))
        img_UR = tf.slice(field, tf.stack([0, 0, N_half]), tf.stack([B, M_half, N - N_half]))
        img_LL = tf.slice(field, tf.stack([0, M_half, 0]), tf.stack([B, M - M_half, N_half]))
        img_LR = tf.slice(field, tf.stack([0, M_half, N_half]), tf.stack([B, M - M_half, N - N_half]))

        return tf.concat([tf.concat([img_LR, img_LL], 2), tf.concat([img_UR, img_UL], 2)], 1)

    def tf_ifft_shift_2d(self, field):

        B, M, N = self.B, self.M, self.N
        def fM0(): return M/2
        def fM1(): return tf.floor(M/2)
        M_half = tf.to_int32(tf.cond(tf.equal(tf.mod(M, 2),0), fM0, fM1))
        def fN0(): return N/2
        def fN1(): return tf.floor(N/2)
        N_half = tf.to_int32(tf.cond(tf.equal(tf.mod(N, 2),0), fN0, fN1))

        N = tf.to_int32(N)
        M = tf.to_int32(M)
        img_UL = tf.slice(field, tf.stack([0, 0, 0]), tf.stack([B, M_half, N_half]))
        img_UR = tf.slice(field, tf.stack([0, 0, N_half]), tf.stack([B, M_half, N - N_half]))
        img_LL = tf.slice(field, tf.stack([0, M_half, 0]), tf.stack([B, M - M_half, N_half]))
        img_LR = tf.slice(field, tf.stack([0, M_half, N_half]), tf.stack([B, M - M_half, N - N_half]))
        return tf.concat([tf.concat([img_LR, img_LL], 2), tf.concat([img_UR, img_UL], 2)], 1)

    def tf_FSPAS_FFT(self, field, wlength, z, dx, dy, ridx, theta_max):
    
        """
        Angular Spectrum Propagation of Coherent Wave Fields
        with optional filtering
        
        INPUTS : 
            U, wave-field in space domain
            wlenght : wavelength of the optical wave
            z : distance of propagation
            dx,dy : sampling intervals in space
            M,N : Size of simulation window
            theta0 : Optional BAndwidth Limitation in DEGREES 
                (if no filtering is desired, only EVANESCENT WAVE IS FILTERED)
        
        OUTPUT : 
            output, propagated wave-field in space domain
        
        """
        
        B, M, N = self.B, self.M, self.N
        output_shape = tf.stack([B,tf.to_int32(M),tf.to_int32(N)])
        wlengtheff = wlength/ridx
        dfx = 1/dx/M
        dfy = 1/dy/N
        fx = tf.expand_dims((tf.range(M,dtype=tf.float32)-(M)/2)*dfx,-1)
        fy = tf.expand_dims((tf.range(N,dtype=tf.float32)-(N)/2)*dfy,0)
        fx2 = tf.matmul(fx**2,tf.ones((1,N),dtype=tf.float32))
        fy2 = tf.matmul(tf.ones((M,1),dtype=tf.float32),fy**2)

        # Diffraction limit
        f0 = 1.0/wlengtheff
        Qd = tf.to_float(tf.less((fx2+fy2),(f0**2)))

        # Prop Anti-aliasing
        Qbw = adaptive_bandlimit(fx,fy,z,dfx,dfy,wlengtheff,theta_max)

        Q = Qd*Qbw
        W = Q*(fx2+fy2)*(wlengtheff**2)

        Hphase = 2*np.pi/wlengtheff*z*((tf.ones((M,N))-W)**(0.5))
        HFSP = tf.complex(Q*tf.cos(Hphase),Q*tf.sin(Hphase))
        ASpectrum = tf.fft2d(field)
        ASpectrum = self.tf_fft_shift_2d(ASpectrum)    
        ASpectrum_z = self.tf_ifft_shift_2d(tf.multiply(HFSP,ASpectrum))
        output = tf.ifft2d(ASpectrum_z)

        Qd = tf.to_complex64(Qd)
        Q = tf.to_complex64(Q)
        unitary_prop_cnst = tf.divide(tf.reduce_sum(tf.square(tf.abs(ASpectrum)),axis=[1,2],keepdims=True),tf.reduce_sum(tf.square(tf.abs(ASpectrum*Qd)),axis=[1,2],keepdims=True))
        unitary_cnst = tf.complex(tf.sqrt(unitary_prop_cnst),tf.zeros_like(unitary_prop_cnst))
        output = tf.multiply(output,unitary_cnst)
        output = tf.slice(output, tf.stack([0, 0, 0]), output_shape)
        scattered_pwr = (tf.reduce_sum(tf.square(tf.abs(unitary_cnst*ASpectrum*Qd*(Qd-Q))),axis=[1,2],keepdims=True))/M/N
        return output, scattered_pwr

class NHW_FSPAS_SDFT:

    def __init__(self, field, wlength, z, dx, dy, ridx,theta_max):

        self.B, self.M, self.N = tf.shape(field)[0],tf.to_float(tf.shape(field)[1]),tf.to_float(tf.shape(field)[2])
        self.output, self.scattered_pwr = self.tf_FSPAS(field, wlength, z, dx, dy, ridx,theta_max)

    def tf_udft2(self, field):
    
        M,N = self.M, self.N
        dividend = tf.complex(tf.sqrt(M*N),tf.zeros_like(M))
        out = tf.fft2d(field)/dividend
        return out 

    def tf_uidft2(self, field):
    
        M,N = self.M, self.N 
        multiplicative_factor = tf.complex(tf.sqrt(M*N),tf.zeros_like(M))
        out = tf.ifft2d(field)*multiplicative_factor
        return out

    def tf_sdft2(self, field):
    
        M,N = self.M, self.N
        x = tf.expand_dims(tf.range(M,dtype=tf.float32),-1)
        y = tf.expand_dims(tf.range(N,dtype=tf.float32),0)
        xphase = tf.matmul(np.pi*(M-1)/M*x,tf.ones((1,N),dtype=tf.float32))
        yphase = tf.matmul(tf.ones((M,1),dtype=tf.float32),np.pi*(N-1)/N*y)
        xyphase = xphase+yphase
        exy = tf.complex(tf.cos(xyphase),tf.sin(xyphase))
        exy = tf.cast(exy,dtype=tf.complex64)
        phaseterm = tf.complex(tf.cos(-np.pi*((M-1)**2)/2/M-np.pi*((N-1)**2)/2/N),\
                               tf.sin(-np.pi*((M-1)**2)/2/M-np.pi*((N-1)**2)/2/N))
        phaseterm = tf.cast(phaseterm,dtype=tf.complex64)
        phaseterm = tf.multiply(phaseterm,exy)
        output = tf.multiply(phaseterm,self.tf_udft2(tf.multiply(field,exy)))
        return output

    def tf_sidft2(self, field):
        
        M,N = self.M, self.N
        x = tf.expand_dims(tf.range(M,dtype=tf.float32),-1)
        y = tf.expand_dims(tf.range(N,dtype=tf.float32),0)
        xphase = tf.matmul(-np.pi*(M-1)/M*x,tf.ones((1,N),dtype=tf.float32))
        yphase = tf.matmul(tf.ones((M,1),dtype=tf.float32),-np.pi*(N-1)/N*y)
        xyphase = xphase+yphase
        exy = tf.complex(tf.cos(xyphase),tf.sin(xyphase))
        exy = tf.cast(exy,dtype=tf.complex64)
        phaseterm = tf.complex(tf.cos(np.pi*((M-1)**2)/2/M+np.pi*((N-1)**2)/2/N),\
                               tf.sin(np.pi*((M-1)**2)/2/M+np.pi*((N-1)**2)/2/N))
        phaseterm = tf.cast(phaseterm,dtype=tf.complex64)
        phaseterm = tf.multiply(phaseterm,exy)
        output = tf.multiply(phaseterm,self.tf_uidft2(tf.multiply(field,exy)))
        return output

    def tf_FSPAS(self, field,wlength,z,dx,dy,ridx,theta_max):
    
        """
        Angular Spectrum Propagation of Coherent Wave Fields
        with optional filtering
        
        INPUTS : 
            U, wave-field in space domain
            wlenght : wavelength of the optical wave
            z : distance of propagation
            dx,dy : sampling intervals in space
            M,N : Size of simulation window
            theta0 : Optional BAndwidth Limitation in DEGREES 
                (if no filtering is desired, only EVANESCENT WAVE IS FILTERED)
        
        OUTPUT : 
            output, propagated wave-field in space domain
        
        """
        
        B, M, N = self.B, self.M, self.N
        output_shape = tf.stack([B,tf.to_int32(M),tf.to_int32(N)])
        wlengtheff = wlength/ridx
        dfx = 1/dx/M
        dfy = 1/dy/N
        fx = tf.expand_dims((tf.range(M,dtype=tf.float32)-(M-1)/2)*dfx,-1)
        fy = tf.expand_dims((tf.range(N,dtype=tf.float32)-(N-1)/2)*dfy,0)
        fx2 = tf.matmul(fx**2,tf.ones((1,N),dtype=tf.float32))
        fy2 = tf.matmul(tf.ones((M,1),dtype=tf.float32),fy**2)
        
        # Diffraction limit
        f0 = 1.0/wlengtheff
        Qd = tf.to_float(tf.less((fx2+fy2),(f0**2)))

        # Prop Anti-aliasing
        Qbw = adaptive_bandlimit(fx,fy,z,dfx,dfy,wlengtheff,theta_max)

        Q = Qd*Qbw
        W = Q*(fx2+fy2)*(wlengtheff**2)
        
        Hphase = 2*np.pi/wlengtheff*z*(tf.ones((M,N))-W)**(0.5)
        HFSP = tf.complex(Q*tf.cos(Hphase),Q*tf.sin(Hphase))   
        ASpectrum = self.tf_sdft2(field)  
        output = self.tf_sidft2(tf.multiply(ASpectrum,HFSP))
        
        Qd = tf.to_complex64(Qd)
        Q = tf.to_complex64(Q)
        unitary_prop_cnst = tf.divide(tf.reduce_sum(tf.square(tf.abs(ASpectrum)),axis=[1,2],keepdims=True),tf.reduce_sum(tf.square(tf.abs(ASpectrum*Qd)),axis=[1,2],keepdims=True))
        unitary_cnst = tf.complex(tf.sqrt(unitary_prop_cnst),tf.zeros_like(unitary_prop_cnst))
        output = tf.multiply(output,unitary_cnst)
        output = tf.slice(output, tf.stack([0, 0, 0]), output_shape)
        scattered_pwr = tf.reduce_sum(tf.square(tf.abs(unitary_cnst*ASpectrum*Qd*(Qd-Q))),axis=[1,2],keepdims=True)/tf.sqrt(M*N)
        return output, scattered_pwr

class NCHW_FSPAS_FFT:

    def __init__(self, field, wlength, z, dx, dy, ridx,theta_max):

        self.B, self.C, self.M, self.N = tf.shape(field)[0],tf.shape(field)[1],tf.to_float(tf.shape(field)[2]),tf.to_float(tf.shape(field)[3])
        self.output, self.scattered_pwr = self.tf_FSPAS_FFT(field, wlength, z, dx, dy, ridx,theta_max)

    def tf_fft_shift_2d(self, field):

        B, C, M, N = self.B, self.C, self.M, self.N
        def fM0(): return M/2
        def fM1(): return tf.floor(M/2)+1
        M_half = tf.to_int32(tf.cond(tf.equal(tf.mod(M, 2),0), fM0, fM1))
        def fN0(): return N/2
        def fN1(): return tf.floor(N/2)+1
        N_half = tf.to_int32(tf.cond(tf.equal(tf.mod(N, 2),0), fN0, fN1))

        N = tf.to_int32(N)
        M = tf.to_int32(M)
        img_UL = tf.slice(field, tf.stack([0, 0, 0, 0]), tf.stack([B, C, M_half, N_half]))
        img_UR = tf.slice(field, tf.stack([0, 0, 0, N_half]), tf.stack([B, C, M_half, N - N_half]))
        img_LL = tf.slice(field, tf.stack([0, 0, M_half, 0]), tf.stack([B, C, M - M_half, N_half]))
        img_LR = tf.slice(field, tf.stack([0, 0, M_half, N_half]), tf.stack([B, C, M - M_half, N - N_half]))

        return tf.concat([tf.concat([img_LR, img_LL], 3), tf.concat([img_UR, img_UL], 3)], 2)

    def tf_ifft_shift_2d(self, field):

        B, C, M, N = self.B, self.C, self.M, self.N
        def fM0(): return M/2
        def fM1(): return tf.floor(M/2)
        M_half = tf.to_int32(tf.cond(tf.equal(tf.mod(M, 2),0), fM0, fM1))
        def fN0(): return N/2
        def fN1(): return tf.floor(N/2)
        N_half = tf.to_int32(tf.cond(tf.equal(tf.mod(N, 2),0), fN0, fN1))

        N = tf.to_int32(N)
        M = tf.to_int32(M)
        img_UL = tf.slice(field, tf.stack([0, 0, 0, 0]), tf.stack([B, C, M_half, N_half]))
        img_UR = tf.slice(field, tf.stack([0, 0, 0, N_half]), tf.stack([B, C, M_half, N - N_half]))
        img_LL = tf.slice(field, tf.stack([0, 0, M_half, 0]), tf.stack([B, C, M - M_half, N_half]))
        img_LR = tf.slice(field, tf.stack([0, 0, M_half, N_half]), tf.stack([B, C, M - M_half, N - N_half]))
        return tf.concat([tf.concat([img_LR, img_LL], 3), tf.concat([img_UR, img_UL], 3)], 2)

    def adaptive_bandlimit(self,fx,fy,z,dfx,dfy,wlength,theta_max):
        
        FX = tf.matmul(fx,tf.ones_like(fy))
        FY = tf.matmul(tf.ones_like(fx),fy) 
        fx_lim = tf.divide(1.0/wlength,(((2.0*dfx*z)**2+1.0)**0.5))
        fy_lim = tf.divide(1.0/wlength,(((2.0*dfy*z)**2+1.0)**0.5))

        theta_max = theta_max/180.0*np.pi # deg2rad
        fx_lim_t = tf.cast(tf.sin(theta_max)/wlength,tf.float32)
        fy_lim_t = tf.cast(tf.sin(theta_max)/wlength,tf.float32)
        fR = tf.square(fx_lim_t)+tf.square(fy_lim_t)
        Qt = tf.to_float(tf.less((tf.square(FX)+tf.square(FY)),(fR)))

        Q = tf.multiply(tf.to_float(tf.less_equal(tf.expand_dims(tf.square(FX),0)/fx_lim**2+tf.expand_dims(tf.square(FY),0)*wlength**2,1.0)),\
            tf.to_float(tf.less_equal(tf.expand_dims(tf.square(FY),0)/fy_lim**2+tf.expand_dims(tf.square(FX),0)*wlength**2,1.0)))

        return Q


    def tf_FSPAS_FFT(self, field, wlength, z, dx, dy, ridx,theta_max):
    
        """
        Angular Spectrum Propagation of Coherent Wave Fields
        with optional filtering
        
        INPUTS : 
            U, wave-field in space domain
            wlenght : MULTI-wavelengthS in the optical wave
            z : distance of propagation
            dx,dy : sampling intervals in space
            M,N : Size of simulation window
            theta0 : Optional BAndwidth Limitation in DEGREES 
                (if no filtering is desired, only EVANESCENT WAVE IS FILTERED)
        
        OUTPUT : 
            output, propagated wave-field in space domain
        
        """
        
        B, C, M, N = self.B, self.C, self.M, self.N
        output_shape = tf.stack([B, C, tf.to_int32(M),tf.to_int32(N)])
        wlengtheff = wlength/ridx
        dfx = 1/dx/M
        dfy = 1/dy/N
        fx = tf.expand_dims((tf.range(M,dtype=tf.float32)-(M)/2)*dfx,-1)
        fy = tf.expand_dims((tf.range(N,dtype=tf.float32)-(N)/2)*dfy,0)
        fx2 = tf.matmul(fx**2,tf.ones((1,N),dtype=tf.float32))
        fy2 = tf.matmul(tf.ones((M,1),dtype=tf.float32),fy**2)
        
        wlengtheff = tf.expand_dims(tf.expand_dims(wlengtheff,-1),-1)
        # Diffraction limit
        f0 = 1.0/wlengtheff
        Qd = tf.to_float(tf.less(tf.expand_dims((fx2+fy2),0),(f0**2)))

        # Prop Anti-aliasing
        Qbw = self.adaptive_bandlimit(fx,fy,z,dfx,dfy,wlengtheff,theta_max) #!!! CHECK THIS

        Q = Qd*Qbw
            
        W = Q*tf.expand_dims((fx2+fy2),0)*(wlengtheff**2)
        phase_term_sqrt = (tf.ones((C,tf.to_int32(M),tf.to_int32(N)))-W)**(0.5)
        Hphase = 2*np.pi/wlengtheff*z*phase_term_sqrt
        HFSP = tf.complex(Q*tf.cos(Hphase),Q*tf.sin(Hphase))
        ASpectrum = tf.fft2d(field)
        ASpectrum = self.tf_fft_shift_2d(ASpectrum)    
        ASpectrum_z = self.tf_ifft_shift_2d(tf.multiply(HFSP,ASpectrum))
        output = tf.ifft2d(ASpectrum_z)
        
        Qd = tf.to_complex64(Qd)
        Q = tf.to_complex64(Q)
        unitary_prop_cnst = tf.divide(tf.reduce_sum(tf.square(tf.abs(ASpectrum)),axis=[2,3],keepdims=True),tf.reduce_sum(tf.square(tf.abs(ASpectrum*Qd)),axis=[2,3],keepdims=True))
        unitary_cnst = tf.complex(tf.sqrt(unitary_prop_cnst),tf.zeros_like(unitary_prop_cnst))
        output = tf.multiply(output,unitary_cnst)
        output = tf.slice(output, tf.stack([0, 0, 0, 0]), output_shape)
        scattered_pwr = tf.reduce_sum(tf.square(tf.abs(unitary_cnst*ASpectrum*Qd*(Qd-Q))),axis=[2,3],keepdims=True)/M/N
        return output, scattered_pwr
'''
D2NN
'''

import tensorflow as tf
import os
import numpy as np
import tf_waveoptics as tfwo
from tf_waveoptics import ideallens
import custom_neural_networks as cnn
import standard_networks as sn
import custom_functions as cf
import tfcookbook as tfcb

#======================================================================
# SENSOR MODELS
#=======================================================================

@tf.custom_gradient
def intensity(z):
    z_ = tf.conj(z)
    I = tf.abs(z*z_)
    def grad(dy):
        return tf.complex(dy,tf.zeros_like(dy)) * z_
    return I, grad

# DISCRETE DETECTORS
def discrete_detectors(measurement,opt):

    ULcorners = np.copy(opt.pixel_locs)
    kk = 0
    pxl_loc = ULcorners[kk,0:2]
    probs = tf.reduce_sum(tf.slice(measurement,[0,pxl_loc[0],pxl_loc[1]],[opt.batch_sz,opt.pixel_row,opt.pixel_col]),axis=[1,2])
    class_probs = tf.expand_dims(probs,-1)
    for kk in range(1, ULcorners.shape[0]):
        pxl_loc = ULcorners[kk,0:2]
        probs = tf.reduce_sum(tf.slice(measurement,[0,pxl_loc[0],pxl_loc[1]],[opt.batch_sz,opt.pixel_row,opt.pixel_col]),axis=[1,2])
        probs = tf.expand_dims(probs,-1)
        class_probs = tf.concat([class_probs,probs],axis=1)

    return class_probs

# FOCAL-PLANE ARRAY
def focal_plane_array(roi, opt):

    H,W = roi.shape[1], roi.shape[2]
    gapx = int((H-opt.pixel_num_x*opt.pixel_row)//(opt.pixel_num_x-1))+opt.pixel_row
    gapy = int((W-opt.pixel_num_y*opt.pixel_col)//(opt.pixel_num_y-1))+opt.pixel_col
    sensor = tf.layers.average_pooling2d(inputs = roi, pool_size=[opt.pixel_row, opt.pixel_col], strides=(gapx,gapy), padding = 'SAME')

    return sensor

def get_hole_params(opt):

    x = (np.arange(opt.mask_row)-(opt.mask_row-1)/2)*opt.dx
    x = np.reshape(x,[opt.mask_row,1])
    y = (np.arange(opt.mask_col)-(opt.mask_col-1)/2)*opt.dy

    x_num = int(np.ceil(np.sqrt(opt.num_holes)))
    y_num = int(np.ceil(opt.num_holes/x_num))
    x_loc_init = np.linspace(-0.99, 0.99, x_num)
    y_loc_init = np.linspace(-0.99, 0.99, y_num)
    randx_loc, randy_loc = np.meshgrid(x_loc_init, y_loc_init)
    randx_loc = np.reshape(randx_loc,(1,x_num*y_num))[:,0:opt.num_holes]
    randy_loc = np.reshape(randy_loc,(1,x_num*y_num))[:,0:opt.num_holes]
    randx_loc = np.tile(randx_loc, (opt.num_holey_layers,1))
    randy_loc = np.tile(randy_loc, (opt.num_holey_layers,1))
    randx_loc = np.arctanh(randx_loc)
    randy_loc = np.arctanh(randy_loc)
    randx_loc = randx_loc.astype('float32')           
    randy_loc = randy_loc.astype('float32')

    return randx_loc, randy_loc, x, y

def create_src_field(opt,d):

    zz = tf.cast(d,tf.float32)
    x = tf.constant((np.arange(opt.M)-(opt.M-1)/2)*opt.dx,shape=[opt.M,1],dtype=tf.float32)
    y = tf.constant((np.arange(opt.N)-(opt.N-1)/2)*opt.dy,shape=[1,opt.N],dtype=tf.float32)
    r2 = tf.matmul(tf.square(x),tf.ones_like(y))+tf.matmul(tf.ones_like(x),tf.square(y))
    Rz = zz*(1+(opt.zr/zz)**2)
    gouy = tf.atan(zz/opt.zr)
    wz = opt.w0*tf.sqrt(1+(zz/opt.zr)**2)
    k = 2*np.pi/opt.wlength
    amp_term = tf.exp(-r2/(wz**2))
    phase_arg = -(k*zz+k*r2/(2*Rz)-gouy)
    Gz = tf.complex(amp_term*tf.cos(phase_arg),amp_term*tf.sin(phase_arg))

    return Gz

def create_src_field_phole(opt):

    zz = tf.cast(opt.thz_src_dist,tf.float32)
    pr = tf.cast(opt.phole_D/2.0,tf.float32)
    x = tf.constant((np.arange(opt.M)-(opt.M-1)/2)*opt.dx,shape=[opt.M,1],dtype=tf.float32)
    y = tf.constant((np.arange(opt.N)-(opt.N-1)/2)*opt.dy,shape=[1,opt.N],dtype=tf.float32)
    r2 = tf.matmul(tf.square(x),tf.ones_like(y))+tf.matmul(tf.ones_like(x),tf.square(y))
    phole = tf.to_float(tf.less_equal(r2,pr**2))
    # src = create_src_field(opt,15e-2)
    phole_field = tf.complex(phole,tf.zeros_like(phole))
    # phole_field = tf.multiply(phole_field,src)
    Gz = tfwo.NHW_FSPAS_FFT(tf.expand_dims(phole_field,0), opt.wlength, zz, opt.dx, opt.dy, 1.0, opt.theta_max)

    return Gz.output

class SW_Coherent_D2NN:

    def __init__(self, field_amp, field_phase, opt, state, ill_type):

        pad_X = int(opt.M//2-opt.mask_row//2)
        pad_Y = int(opt.N//2-opt.mask_col//2)
        self.mask_paddings = tf.constant([[pad_X,pad_X],[pad_Y,pad_Y]])
        self.phase_conv_coeff = 2*np.pi*(opt.ridx-1.0)/(opt.wlength*1e3) # in mm
        self.absorp_coeff = 2*np.pi*opt.extcoeff/(opt.wlength*1e3)
        
        field_amp, field_phase = self.to_d2nn_field(field_amp,field_phase,opt)
        z_obj = field_phase/self.phase_conv_coeff
        transmission_obj = tf.exp(-self.absorp_coeff*z_obj)
        field_amp = tf.multiply(field_amp,transmission_obj)

        self.field = tf.complex(field_amp*tf.cos(field_phase), field_amp*tf.sin(field_phase))
        if(ill_type==1):
            self.ill_field, ill_mask_amp, ill_mask_phase, ill_mask_thickness = self.illumination_design(opt)
        else:
            self.ill_field = tf.ones_like(self.field)

        self.Ein = tf.reduce_sum(tf.square(tf.abs(self.field*self.ill_field)),axis=[1,2])

        self.mask_amp, self.mask_phase, self.mask_thickness, self.mask_holes = self.d2nn_params(opt)
        output = self.d2nn_model(self.field*self.ill_field, self.mask_amp, self.mask_phase, self.mask_holes, opt, state)
        self.field_op = output[0]
        self.roi_signal = output[1]
        self.leakage = output[2]
        self.absorption = output[3]
        
        if(ill_type==1):
            self.mask_amp.append(ill_mask_amp)
            self.mask_phase.append(ill_mask_phase)
            self.mask_thickness.append(ill_mask_thickness)

    def to_d2nn_field(self, field_amp, field_phase, opt):

        H, W = tf.shape(field_amp)[1], tf.shape(field_amp)[2]
        pad_objx = (opt.M - H) // 2
        pad_objy = (opt.N - W) // 2
        field_amp = tf.pad(field_amp, [(0,0), (pad_objx, pad_objx), (pad_objy, pad_objy)], 'constant')
        field_phase = tf.pad(field_phase, [(0,0), (pad_objx, pad_objx), (pad_objy, pad_objy)], 'constant')

        return field_amp, field_phase

    def illumination_design(self,opt):

        zz = tf.cast(opt.thz_src_dist,tf.float32)
        pr = tf.cast(opt.phole_D/2.0,tf.float32)
        x = tf.constant((np.arange(opt.M)-(opt.M-1)/2)*opt.dx,shape=[opt.M,1],dtype=tf.float32)
        y = tf.constant((np.arange(opt.N)-(opt.N-1)/2)*opt.dy,shape=[1,opt.N],dtype=tf.float32)
        aperture_f = tf.matmul(tf.to_float(tf.less_equal(x,pr)),tf.to_float(tf.less_equal(y,pr)))
        layer_num = 16
        TH = opt.thickness*1e3*aperture_f
        mask_phase, mask_amp = self.mask_init(layer_num, opt)
        mask_phase = mask_phase*aperture_f
        #ABSORPTION MODEL
        rotation = tf.floor(mask_phase/(2*np.pi))
        MP = mask_phase-rotation*2*np.pi
        MP = tf.multiply((tf.sign(MP)+1)/2,MP)-tf.multiply((tf.sign(MP)-1)/2,2*np.pi+MP)+(tf.abs(tf.sign(MP))-1)*np.pi
        mask_thickness = MP/self.phase_conv_coeff
        transmission = tf.exp(-self.absorp_coeff*(mask_thickness+TH))
        mask_amp = tf.multiply(tf.multiply(mask_amp,transmission),aperture_f)

        mod_aperture_f = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
        mod_aperture_f = tfwo.NHW_FSPAS_FFT(tf.expand_dims(mod_aperture_f,0), opt.wlength, opt.thickness, opt.dx, opt.dy, opt.ridx, opt.theta_max)
        ill_field = tfwo.NHW_FSPAS_FFT(mod_aperture_f.output, opt.wlength, zz, opt.dx, opt.dy, 1.0, opt.theta_max)

        return ill_field.output, mask_amp, mask_phase, mask_thickness

    def hole_init(self, masknum, opt):

        if masknum < opt.num_holey_layers:
            randx_loc, randy_loc, x, y = get_hole_params(opt)

            hsx = tf.Variable(tf.constant(opt.hole_size,shape=[opt.num_holes,1],dtype=tf.float32),trainable=opt.learnable_hole_size,name ='mask_holes_widthX')
            hsy = tf.Variable(tf.constant(opt.hole_size,shape=[1,opt.num_holes],dtype=tf.float32),trainable=opt.learnable_hole_size,name ='mask_holes_widthY')
            hcx_var = tf.Variable(tf.constant(randx_loc[masknum,:],shape=[opt.num_holes,1]),name ='mask_holes_centerX')
            hcy_var = tf.Variable(tf.constant(randy_loc[masknum,:],shape=[1,opt.num_holes]),name ='mask_holes_centerY')

            hcx = tf.tanh(hcx_var)*(opt.mask_wx/2-hsx/2)
            hcy = tf.tanh(hcy_var)*(opt.mask_wy/2-hsy/2)
            holes = tf.zeros([opt.mask_row, opt.mask_col], dtype=tf.float32)
            for hh in range(opt.num_holes):

                holes += tf.subtract(tf.sigmoid(12.0/opt.dx*(x-(hcx[hh,0]-hsx[hh,0]/2.0))),tf.sigmoid(12.0/opt.dx*(x-(hcx[hh,0]+hsx[hh,0]/2.0))))\
                *tf.subtract(tf.sigmoid(12.0/opt.dy*(y-(hcy[0,hh]-hsy[0,hh]/2.0))),tf.sigmoid(12.0/opt.dy*(y-(hcy[0,hh]+hsy[0,hh]/2.0))))
                holes = tf.clip_by_value(holes,0.0,1.0)
                holes = tf.nn.relu(holes-0.5)

            pad_X = int(opt.M//2-opt.mask_row//2)
            pad_Y = int(opt.N//2-opt.mask_col//2)
            paddings = tf.constant([[pad_X,pad_X],[pad_Y,pad_Y]])
            holes = tf.pad(holes,paddings)
            holes = tf.clip_by_value(holes,0.0,1.0)
            holes = tf.nn.relu(holes-0.5)
            # holes = tf.clip_by_value(holes,0.0,1.0)

        else:

            holes = tf.zeros([opt.M, opt.N], dtype=tf.float32)

        return holes

    def mask_init(self, masknum, opt):

        #====================================================================================================
        #PHASE INITIALIZATION
        #====================================================================================================
        if opt.phase_mod is True:
            with tf.variable_scope("phase_values"):
                if opt.mask_init_type == 'const':
                    mask_phase_org = tf.get_variable('mask_phase' + str(masknum), initializer=tf.constant(0.5, shape=[opt.mask_row, opt.mask_col]))
                    mask_phase = mask_phase_org * 2 * np.pi
                elif opt.mask_init_type == 'random':
                    mask_phase_org = tf.get_variable('mask_phase' + str(masknum), initializer=tf.random_uniform(shape=[opt.mask_row, opt.mask_col],minval=-0.5,maxval=0.5))
                    mask_phase = mask_phase_org * 2 * np.pi
                elif opt.mask_init_type == 'trained':
                    mask_phase_init = cf.read_mask(opt.mask_path, masknum, 'phase', opt.bestmodel)
                    mask_phase_init = mask_phase_init.astype('float32')
                    mask_phase = tf.get_variable('mask_phase' + str(masknum), 
                        initializer=tf.slice(mask_phase_init,[opt.M//2-opt.mask_row//2,opt.N//2-opt.mask_col//2],[opt.mask_row, opt.mask_col]))
                elif opt.mask_init_type == 'custom':
                    if(masknum==2):
                        lens = ideallens(opt.wlength,3*opt.mask_mask_distance/2,[0.0,0.0],[0.0,0.0],opt.mask_wx*4,opt.dx,opt.dy,opt.M,opt.N)    
                        phase_term = (tf.angle(lens)+np.pi)/2/np.pi
                        mask_phase_org = tf.get_variable('mask_phase' + str(masknum), 
                            initializer=tf.slice(phase_term,[opt.M//2-opt.mask_row//2,opt.N//2-opt.mask_col//2],[opt.mask_row, opt.mask_col]))
                        mask_phase = mask_phase_org * 2 * np.pi
                    else:
                        mask_phase_org = tf.get_variable('mask_phase' + str(masknum), initializer=tf.constant(0.0, shape=[opt.mask_row, opt.mask_col]))
                        mask_phase = mask_phase_org * 2 * np.pi
        else:
            mask_phase = tf.zeros([opt.mask_row, opt.mask_col])
            
        #========================================================================================================
        #AMPLITUDE INITIALIZATION
        #========================================================================================================
        if opt.amp_mod is True:
            with tf.variable_scope("amplitude_values"):
                if opt.mask_init_type == 'trained':
                    mask_amp_init = cf.read_mask(opt.mask_path, masknum, 'phase', opt.bestmodel)
                    mask_amp_init = mask_phase_init.astype('float32')
                    mask_amp = tf.get_variable('mask_amp' + str(masknum), 
                        initializer=tf.slice(mask_amp_init,[opt.M//2-opt.mask_row//2,opt.N//2-opt.mask_row//2],[opt.mask_row, opt.mask_col]))
                else:
                    mask_amp_org = tf.get_variable('mask_amp' + str(masknum), initializer=tf.constant(1.0, shape=[opt.mask_row, opt.mask_col]))    
                    mask_amp = tf.nn.relu(mask_amp_org)
        else:
            mask_amp = tf.ones([opt.mask_row, opt.mask_col])
            
        
        mask_phase = tf.pad(mask_phase,self.mask_paddings)
        mask_amp = tf.pad(mask_amp,self.mask_paddings)
        mask_amp = tf.divide(mask_amp,tf.reduce_max(tf.abs(mask_amp)))

        return mask_phase, mask_amp

    def d2nn_params(self, opt):

        light_path = tf.ones((opt.mask_row,opt.mask_col),dtype=tf.float32)
        light_path = tf.pad(light_path,self.mask_paddings)

        d2nn_mask_amp = []
        d2nn_mask_thickness = []
        d2nn_mask_phase = []
        d2nn_holes = []
        for layer_num in range(opt.num_masks):

            mask_holes = self.hole_init(layer_num, opt)
            mask_phase, mask_amp = self.mask_init(layer_num, opt)

            #ABSORPTION MODEL
            rotation = tf.floor(mask_phase/(2*np.pi))
            MP = mask_phase-rotation*2*np.pi
            MP = tf.multiply((tf.sign(MP)+1)/2,MP)-tf.multiply((tf.sign(MP)-1)/2,2*np.pi+MP)+(tf.abs(tf.sign(MP))-1)*np.pi
            mask_thickness = MP/self.phase_conv_coeff
            transmission = tf.exp(-self.absorp_coeff*(1.0-mask_holes)*(mask_thickness+1e3*opt.thickness*light_path))
            mask_amp = tf.multiply(mask_amp,transmission)
            d2nn_mask_amp.append(mask_amp)
            d2nn_mask_thickness.append(mask_thickness)
            d2nn_mask_phase.append(mask_phase)
            d2nn_holes.append(mask_holes)

        return d2nn_mask_amp, d2nn_mask_phase, d2nn_mask_thickness, d2nn_holes

    def d2nn_model(self, field, d2nn_mask_amp, d2nn_mask_phase, d2nn_holes, opt, state):

        light_path = tf.ones((opt.mask_row,opt.mask_col),dtype=tf.float32)
        light_path = tf.pad(light_path,self.mask_paddings)
        scatter_region = tf.ones_like(light_path)-light_path

        d2nn_absorp_loss = []
        d2nn_light_leakage = []
        for layer_num in range(opt.num_masks):

            with tf.name_scope('d2nn'+str(layer_num)):

                with tf.name_scope('propagation'):
                    field_mask = tfwo.NHW_FSPAS_FFT(field, opt.wlength, opt.distances[layer_num], opt.dx, opt.dy, 1.0, opt.theta_max)

                with tf.name_scope('mask'):
                    mask_amp = d2nn_mask_amp[layer_num]
                    mask_phase = d2nn_mask_phase[layer_num]
                    mask_holes = d2nn_holes[layer_num]
                    final_amp = (1.0-mask_holes)*mask_amp + mask_holes
                    mask = tf.complex(final_amp * tf.cos(mask_phase), final_amp * tf.sin(mask_phase))
                    #-----------------------------------------------------------------
                    Escatter = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field_mask.output)),scatter_region),axis=[1,2],keepdims=True)
                    Epath = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field_mask.output)),light_path),axis=[1,2],keepdims=True)
                    #------------------------------------------------------------------

                field = tf.multiply(field_mask.output, mask)
                Eout = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field)),light_path),axis=[1,2],keepdims=True)
                d2nn_absorp_loss.append(tf.subtract(Epath,Eout))
                field_th = tfwo.NHW_FSPAS_FFT(field, opt.wlength, opt.thickness, opt.dx, opt.dy, opt.ridx, opt.theta_max)
                light_leakage = Escatter+field_mask.scattered_pwr+field_th.scattered_pwr
                d2nn_light_leakage.append(light_leakage)
                field = field_th.output

        # To Output Plane
        with tf.name_scope('d2nn_'):
                
            with tf.name_scope('propagation'):
                field_op = tfwo.NHW_FSPAS_FFT(field, opt.wlength, opt.distances[-1], opt.dx, opt.dy, 1.0, opt.theta_max)
                d2nn_light_leakage.append(field_op.scattered_pwr)

            with tf.name_scope('output_plane'):
                roi_signal = tf.slice(field_op.output,[0,opt.M//2-opt.roi_row//2,opt.N//2-opt.roi_col//2],[opt.batch_sz,opt.roi_row,opt.roi_col])

        d2nn_light_leakage = tf.convert_to_tensor(d2nn_light_leakage)
        d2nn_absorp_loss = tf.convert_to_tensor(d2nn_absorp_loss)

        return field_op.output, roi_signal, d2nn_light_leakage, d2nn_absorp_loss

#========================================================================================
# MULTI-WAVELENGTH
#========================================================================================
class MW_Coherent_D2NN:

    def __init__(self, field_amp, field_phase, opt, state, wlength_dependencies):

        pad_X = int(opt.M//2-opt.mask_row//2)
        pad_Y = int(opt.N//2-opt.mask_col//2)
        self.mask_paddings = tf.constant([[pad_X,pad_X],[pad_Y,pad_Y]])

        self.wlength = wlength_dependencies[0]
        self.ridx = wlength_dependencies[1]
        extcoeff = wlength_dependencies[2]
        power = wlength_dependencies[3]

        phase_conv_coeff = 2*np.pi*(self.ridx-1.0)/(self.wlength*1e3) # in mm
        absorp_coeff = 2*np.pi*extcoeff/(self.wlength*1e3)
        self.phase_conv_coeff = tf.expand_dims(tf.expand_dims(tf.expand_dims(phase_conv_coeff,-1),-1),0)
        self.absorp_coeff = tf.expand_dims(tf.expand_dims(tf.expand_dims(absorp_coeff,-1),-1),0)
        power = tf.expand_dims(tf.expand_dims(tf.expand_dims(power,-1),-1),0)

        field_amp, field_phase = self.to_d2nn_field(field_amp,field_phase,opt)
        field_amp = tf.multiply(field_amp,tf.sqrt(power))
        z_obj = field_phase/self.phase_conv_coeff
        transmission_obj = tf.exp(-self.absorp_coeff*z_obj)
        field_amp = tf.multiply(field_amp,transmission_obj)
        self.field = tf.complex(field_amp*tf.cos(field_phase), field_amp*tf.sin(field_phase))

        self.mask_amp, self.mask_phase, self.mask_thickness = self.d2nn_params(opt)
        output = self.d2nn_model(self.field, opt, state, self.mask_amp, self.mask_phase)
        self.op_int = output[0]
        self.sensor_signal = output[1]
        self.leakage = output[2]
        self.transmission = output[3]

    def to_d2nn_field(self, field_amp, field_phase, opt):

        pad_objx = (opt.M - opt.obj_row) // 2
        pad_objy = (opt.N - opt.obj_col) // 2
        field_amp = tf.pad(field_amp, [(0,0), (pad_objx, pad_objx), (pad_objy, pad_objy)], 'constant')
        field_phase = tf.pad(field_phase, [(0,0), (pad_objx, pad_objx), (pad_objy, pad_objy)], 'constant')
        field_amp = tf.expand_dims(field_amp,1)
        field_phase = tf.expand_dims(field_phase,1)
        field_amp = tf.tile(field_amp,[1,opt.Nf,1,1])
        field_phase = tf.tile(field_phase,[1,opt.Nf,1,1])

        return field_amp, field_phase

    def multiWlen_mask_init(self, masknum, opt):

        if opt.phase_mod is True:
            with tf.variable_scope("thickness_values", reuse=tf.AUTO_REUSE):

                mask_thickness_org = tf.get_variable('mask_thickness' + str(masknum), initializer=tf.constant(0.0, shape=[opt.mask_row, opt.mask_col]))
                mask_thickness = (tf.sin(mask_thickness_org)+1.0)/2.0
                mask_thickness = mask_thickness*1e-3

        if opt.amp_mod is True:
            with tf.variable_scope("amplitude_values", reuse=tf.AUTO_REUSE):
                mask_amp_org = tf.get_variable('mask_amp' + str(masknum), initializer=tf.constant(1.0, shape=[opt.mask_row, opt.mask_col]))    
                mask_amp = tf.nn.relu(mask_amp_org)
        else:
            mask_amp = tf.ones([opt.mask_row, opt.mask_col])

        mask_amp = tf.divide(mask_amp,tf.reduce_max(tf.abs(mask_amp)))
        mask_amp = tf.pad(mask_amp,self.mask_paddings)    
        mask_thickness = tf.pad(mask_thickness,self.mask_paddings)    

        return mask_thickness, mask_amp

    def d2nn_params(self, opt):

        d2nn_mask_amp = []
        d2nn_mask_thickness = []
        d2nn_mask_phase = []
        for layer_num in range(opt.num_masks):

            mask_thickness, mask_amp = self.multiWlen_mask_init(layer_num, opt)
            mask_phase = mask_thickness*self.phase_conv_coeff
            transmission = tf.exp(-self.absorp_coeff*mask_thickness)
            mask_amp = tf.multiply(mask_amp,transmission)
            d2nn_mask_amp.append(tf.squeeze(mask_amp))
            d2nn_mask_thickness.append(tf.squeeze(mask_thickness))
            d2nn_mask_phase.append(tf.squeeze(mask_phase))

        d2nn_mask_phase = tf.convert_to_tensor(d2nn_mask_phase)
        d2nn_mask_amp = tf.convert_to_tensor(d2nn_mask_amp)
        d2nn_mask_thickness = tf.convert_to_tensor(d2nn_mask_thickness)
            
        return d2nn_mask_amp, d2nn_mask_phase, d2nn_mask_thickness

    def d2nn_model(self, field, opt, state, d2nn_mask_amp, d2nn_mask_phase):
        
        light_path = tf.ones((opt.mask_row,opt.mask_col),dtype=tf.float32)
        light_path = tf.pad(light_path,self.mask_paddings)
        
        transmission = []
        leakage = []
        # Propagation through D2NN
        for layer_num in range(opt.num_masks):
            with tf.name_scope('hidden' + str(layer_num)):
                
                Ein = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field)),light_path),axis=[1,2,3])
                with tf.name_scope('propagation'):
                    field_z = tfwo.NCHW_FSPAS_FFT(field, self.wlength, opt.mask_mask_dist, opt.dx, opt.dy, 1.0, opt.theta_max)
                    field_mask = tfwo.NCHW_FSPAS_FFT(field_z.output, self.wlength, opt.thickness, opt.dx, opt.dy, self.ridx, opt.theta_max)

                with tf.name_scope('mask'):

                    Epath = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field_mask.output)),light_path),axis=[1,2,3])
                    light_leakage = tf.subtract(Ein,Epath)
                    leakage.append(light_leakage)
                    #------------------------------------------------------------------
                    mask_amp = tf.slice(d2nn_mask_amp,[layer_num,0,0,0],[1,opt.Nf,opt.M,opt.N])
                    mask_phase = tf.slice(d2nn_mask_phase,[layer_num,0,0,0],[1,opt.Nf,opt.M,opt.N])
                    mask = tf.complex(mask_amp* tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                    
                field = tf.multiply(field_mask.output, mask)
                Eout = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field)),light_path),axis=[1,2,3])
                field_transmission = tf.divide(Eout,Epath)
                transmission.append(field_transmission)

        # D2NN_to_sensor
        with tf.name_scope('last'):
                
            Ein = tf.reduce_sum(tf.multiply(tf.square(tf.abs(field)),light_path),axis=[1,2,3])
            with tf.name_scope('propagation'):
                field_op = tfwo.NCHW_FSPAS_FFT(field, self.wlength, opt.mask_sensor_dist, opt.dx, opt.dy, 1.0, opt.theta_max)

            with tf.name_scope('output_plane'):
                op_int = tf.square(tf.abs(field_op.output))
                sensor_signal = tf.slice(op_int,[0,0,opt.M//2-opt.sensor_row//2,opt.N//2-opt.sensor_col//2],[opt.batch_sz,opt.Nf,opt.sensor_row,opt.sensor_col])
                sensor_signal = tf.reduce_sum(sensor_signal,axis=[2,3])
                # sensor_signal = discrete_detectors(roi,opt)
                Epath = tf.reduce_sum(sensor_signal,axis=[1])
                light_leakage = tf.subtract(Ein,Epath)
                leakage.append(light_leakage)

        leakage = tf.convert_to_tensor(leakage)
        transmission = tf.convert_to_tensor(transmission)    

        return op_int, sensor_signal, leakage, transmission
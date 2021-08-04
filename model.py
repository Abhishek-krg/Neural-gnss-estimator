import tensorflow as tf
import numpy as np
import tf_ahrs
from ahrs.common.orientation import am2q
from tqdm.notebook import tqdm

configs={'time_bias_length':5,
         'sampling_freq':300, 
         'madgwick_gain':0.041,
         'use_kalman_smoothing':False, # to be only used in inference
         'precision':'float64'}

samples_per_batch=configs['sampling_freq']
Batch_size=1
Dtype=configs['precision']
tf.keras.backend.set_floatx(configs['precision'])
total_devices=7
Re=6371000.79
pi=3.141592653589793
rad=180/pi

device_maps={
 'SamsungS20Ultra':0,
 'Pixel4XLModded':1,
 'Mi8':2,
 'Pixel4XL':3,
 'Pixel5':4,
 'Pixel4Modded':5,
 'Pixel4':6}



class bias_time(tf.keras.layers.Layer):
    """
         ________________________________________________________________________________________________________________
        |                                                                                                                |
        |   This is the bias time layer for computing added time bias :                                                  |
        |       strides over the samples with specified time_bias_length and removes the time dependent sensor biases    |
        |                                                                                                                |
        |       Inputs :                                                                                                 |
        |           inputs :  of shape (?,time_frames,3)                                                                 |
        |           states :  [h,c] states of previously computed step                                                   |
        |       Returns                                                                                                  |
        |           output : with shape (?,time_frames,3)                                                                |
        |________________________________________________________________________________________________________________|
    """
    def __init__(self,time_bias_length=5,layer_name=None):
        super(bias_time,self).__init__(name=layer_name)
        self.input_dim=time_bias_length
        self.bt=tf.keras.layers.LSTM(3,return_sequences=True,return_state=True,input_shape=(input_dim,3))
    def call(self,inputs,states):
        shape=int(samples_per_batch/self.input_dim)
        out_batch=tf.TensorArray(dtype=Dtype, size=inputs.shape[0],element_shape=(samples_per_batch,3))
        states=[tf.reshape(i,[1,3]) for i in states]
        for b in range(inputs.shape[0]):
            out=tf.TensorArray(dtype=Dtype, size=shape,element_shape=(5,3))
            inp=tf.reshape(inputs[b],[shape,self.input_dim,3]) # mini batched input
            for i in range(shape):
                output,h,c=self.bt(tf.reshape(inp[i],[1,self.input_dim,3]),initial_state=states)
                states=[h,c]
                out=out.write(i,tf.reshape(output,[self.input_dim,3]))
            out_batch=out_batch.write(b,tf.reshape(out.stack(),[shape*self.input_dim,3]))
        return out_batch.stack(),states


class madgwick_filter(tf.keras.layers.Layer):
    """
         _________________________________________________________________________________________________________
        |                                                                                                         |
        |   Standard Madgwick filter implements python ahrs in tf:                                       |
        |       Computes 3D rotation in terms of Quaternions given the acc, gyr and mag readings                  |
        |                                                                                                         |
        |           q[t] = madgwick_filter(q[t-1], acc[t], gyr[t], mag[t])                                        |
        |                                                                                                         |
        |       This is a standard function implemented with batch processing and is tf-graph op compatible       |
        |            inputs :                                                                                     |
        |               q[t-1] : An initial quaternion for time t upon which further quaternions are built        |
        |               acc[t] : accelerometer readings for time stamp t                                          |
        |               gyr[t] : gyro readings for timestamp t                                                    |
        |               mag[t] : magnetometer readings for time stamp t                                           |
        |            return:                                                                                      |
        |               q[t]   : quaternion for timestamp t                                                       |
        |_________________________________________________________________________________________________________|
    """
    def __init__(self,gain=0.033,freq=300):
        super(madgwick_filter,self).__init__(name="Madgick_filter")
        # madgwick assumes acceleration due to gravity on z-axis
    def call(self,acc,gyr,mag,q0,**kwargs):
        Q_batch=tf.TensorArray(dtype=Dtype, size=acc.shape[0],element_shape=(samples_per_batch,4))
        q0=q0/tf.linalg.norm(q0) if q0.shape[0] is not None else tf.convert_to_tensor([0.,0.,0.,0.],dtype=Dtype)
        
        for b in range(acc.shape[0]):
            Q=tf.TensorArray(dtype=Dtype, size=samples_per_batch,element_shape=(4,))
            Q=Q.write(0,tf_ahrs.Madgwick_tf(q0,gyr[b][0],acc[b][0],mag[b][0],gain=configs['madgwick_gain'],freq=configs['sampling_freq']))
            for t in range(1,samples_per_batch):
                Q=Q.write(t,tf_ahrs.Madgwick_tf(Q.read(t-1),gyr[b][t],acc[b][t],mag[b][t],gain=configs['madgwick_gain'], freq=configs['sampling_freq']))
            Q=Q.stack()
            q0=Q[-1]
            Q_batch=Q_batch.write(b,Q)
        return Q_batch.stack()


class compute_accel(tf.keras.layers.Layer):
    """
         _______________________________________________________________________________________________________________
        |                                                                                                               |
        |   This layer computes the acceleration vectors in north and east directions given quaternions and acc :       |
        |                                                                                                               |
        |       quaternion -> Direction Cosine matrix(DCM)                                                              |
        |       ACC[E,N,U]=matmul( ACC[X,Y,Z], (DCM-Inverse) )                                                          |
        |                                                                                                               |
        |       Inputs :                                                                                                |
        |           Q  :  A collection of quaternions                                                                   |
        |           Acc:  coresponding acc vectors in x, y, z axes                                                      |
        |       returns:                                                                                                |
        |           ACC_E, ACC_N                                                                                        |
        |_______________________________________________________________________________________________________________|
    """
    def __init__(self):
        super(compute_accel,self).__init__()
    def call(self,Q,acc_corr,**kwargs):
        R=tf.TensorArray(dtype=Dtype, size=Q.shape[0],element_shape=(samples_per_batch,1,3))
        for batch in range(Q.shape[0]):
            r=tf_ahrs.dcm(Q[batch])
            r=tf.linalg.inv(r)
            accel=tf.TensorArray(dtype=Dtype, size=samples_per_batch,element_shape=(1,3))
            for i in range(samples_per_batch):
                accel=accel.write(i,tf.linalg.matmul([acc_corr[batch][i]],r[i]))
            R=R.write(batch,accel.stack())
        R=R.stack()
        return R

class estimator_outputs(tf.keras.layers.Layer):
    """
         _________________________________________________________________________________________________________
        |                                                                                                         |
        |   The output layer for gnss estimator models :                                                          |
        |       Computes latitude and longitude given acc data, time and prior co-ordinates                       |
        |                                                                                                         |
        |       [lat(t),lon(t)]=[lat(t-1),lon(t-1)]+0.5/Re[acc_E*delta_tsquared,acc_N*delta_tsquared]             |
        |           computed d=0.5*A*delta_tsquared and d=Re*theta or Re*phi => theta or phi=d/Re                 |
        |_________________________________________________________________________________________________________|
    """
    def __init__(self):
        super(estimator_outputs,self).__init__()
    def call(self,R,delta_t,theta,phi):
        ACC_E=R[:,:,:,0]
        ACC_N=R[:,:,:,1]
        delta_tsquared=tf.math.square(delta_t)
        output=tf.TensorArray(dtype=Dtype, size=delta_t.shape[0],element_shape=(2,))
        for b in range(delta_t.shape[0]):
            delta_theta=tf.linalg.matmul(ACC_E[b],delta_tsquared[b],transpose_a=True)
            delta_phi=tf.linalg.matmul(ACC_N[b],delta_tsquared[b],transpose_a=True)
            output=output.write(b,(tf.convert_to_tensor([theta[b],phi[b]])+float(0.5/Re)*tf.convert_to_tensor([delta_theta[0][0],delta_phi[0][0]])))
        return output.stack()


class norm_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(norm_layer,self).__init__()
    def call(self,inputs):
        maxd=tf.math.reduce_max(tf.math.reduce_max(inputs,axis=0),axis=0)
        mind=tf.math.reduce_min(tf.math.reduce_min(inputs,axis=0),axis=0)
        return (inputs-mind)/(maxd-mind),mind,maxd

class denorm_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(denorm_layer,self).__init__()
    def call(self,inputs,mind,maxd):
        return inputs*(maxd-mind)+mind


class gnss_estimator:
    @staticmethod
    def BuildSimpleCorrection(): 
        acc=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=8,dtype=Dtype,name="acc_uncal")
        gyr=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=8,dtype=Dtype,name="gyr_uncal")
        mag=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=8,dtype=Dtype,name="mag_uncal")
        delta_t=tf.keras.Input(shape=(samples_per_batch,1),batch_size=8,dtype=Dtype,name="delta_t")
        theta=tf.keras.Input(shape=(),batch_size=8,dtype=Dtype,name="latitude")
        phi=tf.keras.Input(shape=(),batch_size=8,dtype=Dtype,name="longitude")
        q0=tf.keras.Input(shape=(4),dtype=Dtype,name="q0")

        Q=madgwick_filter()(acc,gyr,mag,q0) # madgwick sampling freq and gain are hyper-parameter and can be assigned while calling model 

        ACC_E,ACC_N=compute_accel()(Q,acc)

        output=estimator_outputs()(ACC_E,ACC_N,delta_t,theta,phi)

        return tf.function(func=tf.keras.Model(inputs=[acc,gyr,mag,delta_t,theta,phi,q0],outputs=[output,Q[-1][-1]]))


    @staticmethod
    def BuildTimeBiasCorrection():
        acc=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="acc_uncal")
        gyr=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="gyr_uncal")
        mag=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="mag_uncal")
        delta_t=tf.keras.Input(shape=(samples_per_batch,1),batch_size=1,dtype=Dtype,name="delta_t")
        theta=tf.keras.Input(shape=(),batch_size=1,dtype=Dtype,name="latitude")
        phi=tf.keras.Input(shape=(),batch_size=1,dtype=Dtype,name="longitude")

        acc_states=[tf.keras.Input(shape=(1,3),dtype=Dtype),tf.keras.Input(shape=(1,3),dtype=Dtype)]
        gyr_states=[tf.keras.Input(shape=(1,3),dtype=Dtype),tf.keras.Input(shape=(1,3),dtype=Dtype)]
        mag_states=[tf.keras.Input(shape=(1,3),dtype=Dtype),tf.keras.Input(shape=(1,3),dtype=Dtype)]
        
        device=tf.keras.Input(shape=(),dtype=Dtype,name="device")
        q0=tf.keras.Input(shape=(4),dtype=Dtype,name="q0")

        interactions=tf.reshape(tf.keras.layers.Embedding(total_devices,2,name="device_embeds")(device),[1,2])
        acc_embeds=tf.keras.layers.Dense(3,activation=None,name='acc_embeds')(interactions)
        gyr_embeds=tf.keras.layers.Dense(3,activation=None,name='gyr_embeds')(interactions)
        mag_embeds=tf.keras.layers.Dense(3,activation=None,name='mag_embeds')(interactions)
        
        acc_corr=tf.keras.layers.Subtract()([acc,tf.broadcast_to(acc_embeds,acc.shape)])
        gyr_corr=tf.keras.layers.Subtract()([gyr,tf.broadcast_to(gyr_embeds,gyr.shape)])
        mag_corr=tf.keras.layers.Subtract()([mag,tf.broadcast_to(mag_embeds,mag.shape)])

        acc_corr,acc_min,acc_max=norm_layer()(acc_corr)
        acc_corr,acc_states_out=bias_time(layer_name="acc_bias")(acc_corr,acc_states)
        acc_corr=denorm_layer()(acc_corr,acc_min,acc_max)

        gyr_corr,gyr_min,gyr_max=norm_layer()(gyr_corr)
        gyr_corr,gyr_states_out=bias_time(layer_name="gyr_bias")(gyr_corr,gyr_states)
        gyr_corr=denorm_layer()(gyr_corr,gyr_min,gyr_max)

        mag_corr,mag_min,mag_max=norm_layer()(mag_corr)
        mag_corr,mag_states_out=bias_time(layer_name="mag_bias")(mag_corr,mag_states)
        mag_corr=denorm_layer()(mag_corr,mag_min,mag_max)

        Q=madgwick_filter()(acc_corr,gyr_corr,mag_corr,q0)

        ACC_E,ACC_N=compute_accel()(Q,acc_corr)

        output=estimator_outputs()(ACC_E,ACC_N,delta_t,theta,phi)

        return tf.keras.Model(inputs=[acc,gyr,mag,delta_t,theta,phi,acc_states,gyr_states,mag_states,device,q0],outputs=[output,acc_states_out,gyr_states_out,mag_states_out,Q[-1][-1]])

    @staticmethod
    def BuildConvCorrection():
        acc=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="acc_uncal")
        gyr=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="gyr_uncal")
        mag=tf.keras.Input(shape=(samples_per_batch,3,),batch_size=1,dtype=Dtype,name="mag_uncal")
        delta_t=tf.keras.Input(shape=(samples_per_batch,1),batch_size=1,dtype=Dtype,name="delta_t")
        theta=tf.keras.Input(shape=(),batch_size=1,dtype=Dtype,name="latitude")
        phi=tf.keras.Input(shape=(),batch_size=1,dtype=Dtype,name="longitude")
        device=tf.keras.Input(shape=(),batch_size=1,dtype=Dtype,name="device")
        q0=tf.keras.Input(shape=(4),dtype=Dtype,name="q0")

        interactions=tf.reshape(tf.keras.layers.Embedding(total_devices,2,name="device_embeds")(device),[1,2])
        acc_embeds=tf.keras.layers.Dense(3,activation=None,name='acc_embeds')(interactions)
        gyr_embeds=tf.keras.layers.Dense(3,activation=None,name='gyr_embeds')(interactions)
        mag_embeds=tf.keras.layers.Dense(3,activation=None,name='mag_embeds')(interactions)
        
        acc_corr=tf.keras.layers.Subtract()([acc,tf.broadcast_to(acc_embeds,acc.shape)])
        gyr_corr=tf.keras.layers.Subtract()([gyr,tf.broadcast_to(gyr_embeds,gyr.shape)])
        mag_corr=tf.keras.layers.Subtract()([mag,tf.broadcast_to(mag_embeds,mag.shape)])

        acc_corr,acc_min,acc_max=norm_layer()(acc_corr)
        acc_corr=tf.keras.layers.Conv1D(10,5,strides=1,padding='same')(acc_corr)
        acc_corr=tf.keras.layers.Conv1DTranspose(3,5,strides=1,padding='same')(acc_corr)
        acc_corr=denorm_layer()(acc_corr,acc_min,acc_max)

        gyr_corr,gyr_min,gyr_max=norm_layer()(gyr_corr)
        gyr_corr=tf.keras.layers.Conv1D(10,5,strides=1,padding='same')(gyr_corr)
        gyr_corr=tf.keras.layers.Conv1DTranspose(3,5,strides=1,padding='same')(gyr_corr)
        gyr_corr=denorm_layer()(gyr_corr,gyr_min,gyr_max)

        mag_corr,mag_min,mag_max=norm_layer()(mag_corr)
        mag_corr=tf.keras.layers.Conv1D(10,5,strides=1,padding='same')(mag_corr)
        mag_corr=tf.keras.layers.Conv1DTranspose(3,5,strides=1,padding='same')(mag_corr)
        mag_corr=denorm_layer()(mag_corr,mag_min,mag_max)

        Q=madgwick_filter()(acc_corr,gyr_corr,mag_corr,q0)

        R=compute_accel()(Q,acc_corr)

        output=estimator_outputs()(R,delta_t,theta,phi)

        return tf.keras.Model(inputs=[acc,gyr,mag,delta_t,theta,phi,device,q0],outputs=[output,Q[-1][-1]])


def haversine_loss(d1,d2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    D=tf.convert_to_tensor([d1,d2])
    D=rad*D
    d1=D[0];d2=D[1]
    lat1 = d1[:,0] ; lon1 = d1[:,1]
    lat2 = d2[:,0] ; lon2 = d2[:,1]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = tf.math.sin(dlat/2.0)**2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon/2.0)**2

    c = 2 * tf.math.asin(a**0.5)
    dist = Re * c
    return dist

def gen_q0(acc,mag):
    return am2q(acc,mag)

def train_model(collection,epoch=1):
    for f in tqdm(collection,desc="dataset collection"):
        dataset=np.load("decimeter/train/"+f+'.npy',allow_pickle=True)
        q=model.gen_q0(dataset[0]['q0'][0],dataset[0]['q0'][1]) # create a new Quaternion when a new collection is loaded 
        device=model.device_maps[str(f.split('_')[-1])]
        acc=[];gyr=[];mag=[];delta_t=[];lat=[];lon=[];gt_lat=[];gt_lon=[]
        for d in tqdm(np.nditer(dataset,flags=['refs_ok']),desc="processing data"):
            xt=d.item()['readings']
            yt=d.item()['ground_truth']
            xt=process_samples(xt) # pre-process data to fixed sampling interval default 300
            if (xt!=None):
                acc.append(xt[0])
                gyr.append(xt[1])
                mag.append(xt[2])
                delta_t.append(xt[3])
                lat.append(xt[4])
                lon.append(xt[5])
                gt_lat.append(yt[0])
                gt_lon.append(yt[1])
        dataset=tf.data.Dataset.from_tensor_slices({'acc':acc,'gyr':gyr,'mag':mag,'delta_t':delta_t,'lat':lat,'lon':lon,'gt_lat':gt_lat,'gt_lon':gt_lon})
        dataset=dataset.batch(4)
        @tf.function
        def train_f(dataset,q,device):
            metrics=tf.TensorArray(dtype=tf.float64,size=tf.cast(dataset.cardinality(),dtype=tf.int32))
            for train_step ,d in enumerate(dataset):
                with tf.GradientTape() as g:
                    g.watch(m.trainable_weights)
                    out,q=m([d['acc'],d['gyr'],d['mag'],d['delta_t'],d['lat'],d['lon'],device,q],training=True)
                    loss=hv_loss(out,tf.concat([[d['gt_lat']],[d['gt_lon']]],1))
                    metrics=metrics.write(tf.cast(train_step,dtype=tf.int32),loss)
                grads=g.gradient(loss,m.trainable_weights)
                optimizer.apply_gradients(zip(grads,m.trainable_weights))
            return metrics.stack()
        
        metrics=train_f(dataset,q,device)
        print("Mean loss for "+f+" : ",tf.math.reduce_mean(metrics))
                

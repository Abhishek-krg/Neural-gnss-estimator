import numpy as np

def sensors_pp(arr,to_size):
    if(arr.shape[0]-to_size>to_size):
        s=int(arr.shape[0]/2)
        d=[2*i+1 for i in range(s)]
        arr=np.delete(arr,d,0)
        return sensors_pp(arr,to_size)
    else:
        s=arr.shape[0]-to_size
        d=[2*i+1 for i in range(s)]
        arr=np.delete(arr,d,0)
        return arr

def delta_tpp(arr,to_size):
    if(arr.shape[0]-to_size>to_size):
        s=int(arr.shape[0]/2)
        d=np.array([2*i+1 for i in range(s)])
        arr[d-1]=arr[d-1]+arr[d]
        arr=np.delete(arr,d,0)
        return delta_tpp(arr,to_size)
    else:
        s=arr.shape[0]-to_size
        d=np.array([2*i+1 for i in range(s)])
        arr[d-1]=arr[d-1]+arr[d]
        arr=np.delete(arr,d,0)
        return arr

def process_samples(xt,to_shape=300):
    if(xt==0 or xt[0].shape[0]<to_shape):
        return None
    else:
        return [ sensors_pp(xt[0],to_shape) , sensors_pp(xt[1],to_shape) , sensors_pp(xt[2],to_shape) , delta_tpp(xt[3],to_shape).reshape(to_shape,1)*1e-3 , xt[4] , xt[5] ]


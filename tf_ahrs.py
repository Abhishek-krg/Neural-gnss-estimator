import tensorflow as tf

def dcm(q):
    if q.shape[-1]!=4:
        raise ValueError("Quaternion must be of the form (4,) or (N, 4)")
    if tf.rank(q)>1:
        q /= tf.linalg.norm(q, axis=1)[:, None]
        R00 = 1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)
        R10 = 2.0*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3])
        R20 = 2.0*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
        R01 = 2.0*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])
        R11 = 1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)
        R21 = 2.0*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3])
        R02 = 2.0*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2])
        R12 = 2.0*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1])
        R22 = 1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)

        R=tf.TensorArray(dtype=tf.float64,size=q.shape[0],element_shape=(3,3))
        for i in range(q.shape[0]):
            R=R.write(i,[[R00[i],R01[i],R02[i]],
                        [R10[i],R11[i],R12[i]],
                        [R20[i],R21[i],R22[i]]])
        return R.stack()

def q_conj(q: tf.Tensor) -> tf.Tensor:
    if len(q.shape)>2 or q.shape[-1]!=4:
        raise ValueError("Quaternion must be of shape (4,) or (N, 4), but has shape {}".format(q.shape))
    return tf.convert_to_tensor([1., -1., -1., -1.],dtype=tf.float64)*tf.convert_to_tensor(q,dtype=tf.float64)

def q_prod(p: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    pq0 = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    pq1 = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    pq2 = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    pq3 = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return tf.convert_to_tensor([pq0,pq1,pq2,pq3],dtype=tf.float64)


def Madgwick_tf(q: tf.Tensor, gyr: tf.Tensor, acc: tf.Tensor, mag: tf.Tensor,gain=0.041,freq=100) -> tf.Tensor:
        '''if gyr is None or not tf.linalg.norm(gyr)>0:
            return q'''
        qDot = 0.5 * q_prod(q, tf.convert_to_tensor([0,gyr[0],gyr[1],gyr[2]],dtype=tf.float64))                           # (eq. 12)
        a_norm = tf.linalg.norm(acc)
        Dt=1/freq
        if a_norm>0:
            a = acc/a_norm
            m = mag/tf.linalg.norm(mag)
            # Rotate normalized magnetometer measurements
            h = q_prod(q, q_prod([0, m[0],m[1],m[2]], q_conj(q)))               # (eq. 45)
            bx = tf.linalg.norm([h[1], h[2]])                       # (eq. 46)
            bz = h[3]
            q= q/tf.linalg.norm(q)
            qw=q[0]; qx=q[1]; qy=q[2]; qz=q[3]
            # Gradient objective function (eq. 31) and Jacobian (eq. 32)
            f = tf.convert_to_tensor([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2],
                          2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                          2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                          2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]],dtype=tf.float64)  # (eq. 31)
            f=tf.reshape(f,[6,1])
            J = tf.convert_to_tensor([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                          [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                          [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                          [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                          [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                          [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx]],dtype=tf.float64) # (eq. 32)
            gradient = tf.linalg.matmul(J,f,transpose_a=True)       # (eq. 34)
            gradient = tf.reshape(gradient,[4])
            gradient /= tf.linalg.norm(gradient)
            qDot -= gain*gradient                              # (eq. 33)
        q += qDot*Dt                                           # (eq. 13)
        q /= tf.linalg.norm(q)
        return q

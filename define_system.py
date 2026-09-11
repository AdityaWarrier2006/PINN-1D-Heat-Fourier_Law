import tensorflow as tf
import numpy as np
import deepxde as dde

import get_data

# extract coordinates (x, t) and corresponding temperature (u_exact)

observe_x = df_sampled[['x', 't']].values
print(observe_x[5]) # print
print(observe_x.shape)

observe_u = df_sampled[['T']].values
print(observe_u[5])
print(observe_u.shape)

# create a PointSet boundary condition for the inverse problem

observe_bc = dde.PointSetBC(observe_x, observe_u, component=0)

def pde(x, y):
    '''
    x: input tensor containing domain coordinates
    x[:,0] - spatial coordinates s
    x[:,1] - time coordinates t

    y: Network prediction for the target variable u(x, t)
    '''
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    # The network has to figure the value of alpha

    '''
    i, j are the indexes of x and y respectively
    dy_t = dy/dt
    dy_xx = d2y/dx2
    '''
    # Force alpha to be positive during gradient updates
    return dy_t - tf.math.abs(alpha) * dy_xx

# Spatial domain: x in [0, 1]
# Time domain: t in [0, 20]
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 20)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

alpha = dde.Variable(0.001) # set an initial value

# Boundary Condtion (Walls at x=0 and x=1 are at 100 degrees)
def boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0.0) or np.isclose(x[0], 1.0))

bc = dde.icbc.DirichletBC(geomtime, lambda x: 100.0, boundary)

def initial_smooth(x):
    # x[:, 0:1] extracts the spatial coordinates.
    # The exponentials create a steep gradient from 100 at the edges down to 25.
    x_pos = x[:, 0:1]
    return 25.0 + 75.0 * (np.exp(-150 * x_pos) + np.exp(-150 * (1 - x_pos)))

ic = dde.icbc.IC(geomtime, initial_smooth, lambda _, on_init: on_init)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_bc],
    num_domain=10000, # Increased for better spatial-temporal resolution
    num_boundary=500, # Increased to strictly enforce wall temperatures
    num_initial=500,  # Increased to capture the steep initial gradient
    train_distribution="Hammersley",
    anchors=observe_x
)

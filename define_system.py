import argparse
import tensorflow as tf
import numpy as np
import deepxde as dde

# Setup Argument Parser
parser = argparse.ArgumentParser(description='Initialize points along the system')
parser.add_argument('--num_domain', type=int, default=10000, help='Number of collocation points')
parser.add_argument('--num_boundary', type=int, default=500, help='Number of boundary points')
parser.add_argument('--num_initial', type=int, default=500, help='Number of initial points')
parser.add_argument('--training_distribution', type=str, default='Hammersley', help='Training distribution')

# Use parse_known_args() so importing in notebooks/kernels won't raise SystemExit
args, _ = parser.parse_known_args()

data = define_system.build_pinn_data(observe_x, observe_u)

# PDE setup & Geometry
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    return dy_t - tf.math.abs(alpha) * dy_xx

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 20)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
alpha = dde.Variable(0.001)

def boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0.0) or np.isclose(x[0], 1.0))

bc = dde.icbc.DirichletBC(geomtime, lambda x: 100.0, boundary)

def initial_smooth(x):
    x_pos = x[:, 0:1]
    return 25.0 + 75.0 * (np.exp(-150 * x_pos) + np.exp(-150 * (1 - x_pos)))

ic = dde.icbc.IC(geomtime, initial_smooth, lambda _, on_init: on_init)

# 2. Function accepting CLI args as default parameter values
def build_pinn_data(
    observe_x, 
    observe_u, 
    geomtime=geomtime, 
    num_domain=args.num_domain, 
    num_boundary=args.num_boundary, 
    num_initial=args.num_initial, 
    training_distribution=args.training_distribution):
  
    observe_bc = dde.icbc.PointSetBC(observe_x, observe_u, component=0)
    
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_bc],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        anchors=observe_x
    )
    return data

# 3. CLI Entry Point (Only runs when executed as `python define_system.py`)
if __name__ == "__main__":
    observe_x, observe_u, observe_bc = get_data.get_data()
    data = build_pinn_data(observe_x, observe_u)
    print("PINN Data initialized successfully via CLI.")

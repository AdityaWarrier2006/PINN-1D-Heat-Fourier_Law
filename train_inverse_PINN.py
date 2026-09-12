import argparse
import deepxde as dde
import tensorflow as tf
import define_system
import get_data

def train_inverse_pinn(
    data,
    alpha,
    layer_sizes=None,
    adam_lr=0.001,
    adam_iters=15000,
    save_path="./models/my_ipinn_model"
):
    """Builds, compiles, and trains a PINN to solve the inverse 1D heat equation."""
    if layer_sizes is None:
        layer_sizes = [2] + [50] * 4 + [1]

    # 1. Define the Neural Network
    net = dde.nn.FNN(layer_sizes, "tanh", "Glorot uniform")
    net.apply_output_transform(lambda x, y: y * 37.5 + 62.5)

    # 2. Compile and Train with Adam
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=adam_lr,
        external_trainable_variables=alpha,
        loss_weights=[1, 1, 1, 100]
    )
    print(f"Starting Adam Optimization (lr={adam_lr}, iters={adam_iters})...")
    model.train(iterations=adam_iters, display_every=1000)

    # 3. Fine-tune with L-BFGS
    model.compile("L-BFGS", external_trainable_variables=alpha)
    print("Starting L-BFGS Optimization...")
    losshistory, train_state = model.train()

    # 4. Print Results & Save
    print("Final alpha value:", alpha.numpy())
    model.save(save_path)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    return model, losshistory, train_state

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Inverse PINN model.")
    parser.add_argument(
        "--lr",
        "--adam-lr",
        dest="adam_lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--iters",
        "--adam-iters",
        dest="adam_iters",
        type=int,
        default=15000,
        help="Number of iterations for Adam optimizer"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/my_ipinn_model",
        help="Directory to save the trained model"
    )
    
    # Use parse_known_args() so running in Colab/Jupyter won't throw errors
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Load data and setup PINN data instance
    observe_x, observe_u, observe_bc = get_data.get_data()
    data = define_system.build_pinn_data(observe_x, observe_u)
    alpha = define_system.alpha

    # Train model
    train_inverse_pinn(
        data=data,
        alpha=alpha,
        adam_lr=args.adam_lr,
        adam_iters=args.adam_iters,
        save_path=args.save_path
    )

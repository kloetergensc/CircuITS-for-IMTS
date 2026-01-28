import argparse
import os
import pdb
import random
import sys
import time
from random import SystemRandom

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.circuits.circuits import CircuITS
from src.utils import load_data
from src.utils.plot import plot_dataset_examples, plot_instance_inference
from src.utils.utils import (
    evaluate_mnll,
    evaluate_njnll,
    get_gpu_memory_allocated,
    sample_based_metrics,
    test_model_integrates_to_one,
    test_sample_likelihood_consistency,
)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="Training Script for Circuits.")

# Task arguments
parser.add_argument("-s", "--seed", default=1, type=int, help="Set the random seed.")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ct", "--cond_time", default=36, type=int, help="Conditioning range in hours")
parser.add_argument("-ft", "--forc_time", default=0, type=int, help="Forecast horizon in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="Number of folds for crossvalidation")
parser.add_argument("--fold", default=0, type=int, help="Fold in crossvalidation")
parser.add_argument("--plot", action="store_true", help="Enable plotting of samples")
parser.add_argument("--test_consistency", action="store_true", help="Test if sampling and liklyhood is consistent")


# Optimizer arguments
parser.add_argument("--patience", default=30, type=int, help="Patience for early stopping")
parser.add_argument("--scheduler_patience", default=5, type=int)
parser.add_argument("-e", "--epochs", default=2000, type=int, help="Maximum epochs")
parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-vbs", "--val_batch_size", default=64, type=int, help="Validation batch size")
parser.add_argument("-lr", "--learn_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("-b", "--betas", default=(0.9, 0.999), type=float, help="Adam betas", nargs=2)
parser.add_argument("-wd", "--weight_decay", default=0.001, type=float, help="Weight decay")
parser.add_argument("-gc", "--gradient_clip", default=None, type=float, help="Gradient Clipping value")
parser.add_argument("--ignore-copula", action="store_true", help="if activated conduct the w/o Gaussian Copula ablation study")
parser.add_argument("--track_sample_based_metrics", action="store_true", help="if true energy score and crps will be evaluated every epoch")
parser.add_argument("--optimize_energy", action="store_true", help="if activate early stopping will be performed based on the energy score")
parser.add_argument("--permute_channels", action="store_true", help="help if True channels will be permuted in some random order")

parser.add_argument("--channel", default=4, type=int, help="only use together with the brownian synthetic dataset")
parser.add_argument("--timesteps", default=50, type=int, help="only use together with the brownian synthetic dataset")
parser.add_argument("--effi", action="store_true")

# Model specific arguments
parser.add_argument("-nh", "--n_heads", default=2, type=int, help="Number of heads for MHA in encoder")
parser.add_argument("-d", "--hidden_dim", default=64, type=int, help="Number of hidden dimensions")
parser.add_argument("--mlp_layers", default=2, type=int, help="Number of layers in MLP for flow_args")
parser.add_argument("--mlp_dim", default=32, type=int, help="Hidden dimension of MLP for flow_args")
parser.add_argument("--flow_layers", default=2, type=int, help="Number of flow layers for flow_args")
parser.add_argument("--flow_hid_dim", default=10, type=int, help="Hidden dimension in flow layers for flow_args")
parser.add_argument("--components", default=2, type=int, help="Number of components for the model")
parser.add_argument("--marginal_flow", default="dsf", type=str, choices=["dsf", "none"], help="Type of flow for marginal distributions")
parser.add_argument("--encoder_type", default="default", type=str, choices=["default", "TF"])

args = parser.parse_args()

print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(args, experiment_id)

if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

optimizer_config = {
    "lr": args.learn_rate,
    # "betas": args.betas,
    "weight_decay": args.weight_decay,
}

task = load_data.load_dataset(args)


train_loader, val_loader, test_loader, test_loader_sampling = load_data.data_loaders(task, args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Logging
logging_dir = f"tensorboard/{args.dataset}/{args.cond_time}-{args.forc_time}/log_dir-{experiment_id}"
os.makedirs(logging_dir, exist_ok=True)
writer = SummaryWriter(logging_dir)

# Checkpointing
chkpoint_path = f"saved_models/{args.cond_time}_{args.forc_time}/{args.dataset}/d={args.hidden_dim}_K={args.components}_FL={args.flow_layers}/circuits{experiment_id}.h5"
# ensure checkpoint directory exists
os.makedirs(os.path.dirname(chkpoint_path), exist_ok=True)
plot_path = f"plots/{args.dataset}/{experiment_id}/"

channel_order = np.arange(task.dataset.shape[-1])
if args.permute_channels:
    np.random.seed(args.seed)
    channel_order = np.random.permutation(channel_order)
print("Channel order for Recursive SPN:", channel_order)

if args.plot:
    plot_dataset_examples(
        path=plot_path,
        dataloader=test_loader_sampling,
        n_channels=task.dataset.shape[-1],
        n_examples=50,
    )

model_config = {
    "n_channels": task.dataset.shape[-1],
    "hidden_dim": args.hidden_dim,
    "n_heads": args.n_heads,
    "flow_args": {
        "context_dim": args.hidden_dim,
        "mlp_layers": args.mlp_layers,
        "mlp_dim": args.mlp_dim,
        "flow_layers": args.flow_layers,
        "flow_hid_dim": args.flow_hid_dim,
    },
    "components": args.components,
    "marginal_flow": args.marginal_flow,
    "device": device,
    "encoder_type": args.encoder_type,
    "channel_order": channel_order,
}


model = CircuITS(**model_config).to(device)

model.zero_grad(set_to_none=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

if not args.ignore_copula:
    model.activate_copula()

from torch.optim import AdamW

nepochs = args.epochs
best_val_loss = 1e8
early_stop = 0


optimizer = AdamW(model.parameters(), **optimizer_config)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.scheduler_patience, factor=0.5, min_lr=0.00001, verbose=True)

for epoch in range(nepochs):
    train_njnll = 0
    count = 0
    model.train()
    ts = time.time()
    for batch in train_loader:
        optimizer.zero_grad()
        obs, mobs, xq, mq, y = (tensor.to(device) for tensor in batch)
        if xq.shape[1] == 0:
            continue
        jnll = model.jnll(obs, mobs, xq, mq, y)
        nsamples = (mq.sum(-1) > 0).sum()
        njnll = jnll / (mq.sum(1) + (mq.sum(1) < 1).to(int))
        train_njnll += njnll[mq.sum(1) > 0].sum()
        count += nsamples
        alloc = get_gpu_memory_allocated()
        njnll[mq.sum(1) > 0].sum().backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
    te = time.time()
    print(f"epoch: {epoch}, train_loss: {train_njnll / count:.6f}, epoch_time: {te - ts:.2f}")
    print(f"Alloc: {alloc:.3f} GB")
    if args.effi:
        sys.exit()
    writer.add_scalar("Loss/train_njnll", train_njnll / count, epoch)

    model.eval()
    with torch.no_grad():
        ts = time.time()
        val_njnll = evaluate_njnll(model=model, loader=val_loader)
        te = time.time()
        if args.track_sample_based_metrics and (epoch % 10) == 0:
            sample_based_metrics_res = sample_based_metrics(model=model, loader=test_loader_sampling)
            mse = sample_based_metrics_res["mse"].item()
            crps = sample_based_metrics_res["crps"].item()
            energy_score = sample_based_metrics_res["energy_score"].item()
            print(f"test_mse: {mse:.6f}")
            print(f"test_crps: {crps:.6f}")
            print(f"test_energy_score: {energy_score:.6f}")
            writer.add_scalar("Metrics/test_mse", mse, epoch)
            writer.add_scalar("Metrics/test_crps", crps, epoch)
            writer.add_scalar("Metrics/test_energy_score", energy_score, epoch)
    print(f"val_loss: {val_njnll.item():.6f}, val_time: {te - ts:.2f}")
    writer.add_scalar("Loss/val", val_njnll, epoch)
    if args.optimize_energy:
        sample_based_metrics_res = sample_based_metrics(model=model, loader=test_loader_sampling)
        energy_score = sample_based_metrics_res["energy_score"].item()
        print(f"valid_energy_score: {energy_score:.6f}")
        writer.add_scalar("energy_score/val", energy_score, epoch)
        val_loss = energy_score
    else:
        val_loss = val_njnll
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            chkpoint_path,
        )
        early_stop = 0
        use_copula = model.use_copula
        print("New best Validation Loss - model checkpoint saved")
    else:
        early_stop += 1

    scheduler.step(val_loss)
    if epoch % 20 == 0 and args.plot:
        batch = next(iter(test_loader_sampling))
        with torch.no_grad():
            plot_instance_inference(
                model=model,
                batch=batch,
                path=f"{plot_path}/EPOCH={epoch}.png",
                n_samples=50,
                model_name=f"CircuITS-{args.components}",
                color="rainbow",
            )
    # check if training should be ended
    if (early_stop == args.patience) or (epoch == nepochs - 1):
        if early_stop == args.patience:
            print(f"Early stopping because of no improvement in val. metric for {args.patience} epochs")
        else:
            print("Completed all the epochs")
        break


chp = torch.load(chkpoint_path)
model.load_state_dict(chp["state_dict"])
model.use_copula = use_copula
with torch.no_grad():
    test_start_time = time.time()
    test_njnll = evaluate_njnll(model=model, loader=test_loader)
    test_end_time = time.time()
    test_mnll = evaluate_mnll(model=model, loader=test_loader)
    writer.add_scalar("test loss", test_njnll, epoch)
    print(f"best_val_loss: {best_val_loss:.6f}")
    print(f"test_time: {test_end_time - test_start_time:.2f}")
    print(f"test_njnll: {test_njnll:.6f}")
    print(f"test_mnll: {test_mnll:.6f}")

    sample_based_metrics_res = sample_based_metrics(model=model, loader=test_loader_sampling)
    mse = sample_based_metrics_res["mse"].item()
    crps = sample_based_metrics_res["crps"].item()
    energy_score = sample_based_metrics_res["energy_score"].item()

    print(f"test_mse: {mse:.6f}")
    print(f"test_crps: {crps:.6f}")
    print(f"test_energy_score: {energy_score:.6f}")

if args.test_consistency:
    test_model_integrates_to_one(model=model)
    test_sample_likelihood_consistency(model=model)

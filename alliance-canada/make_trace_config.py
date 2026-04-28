import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate a diffusion model config for trace datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name under TraceReconstruction-main/Datasets")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length, e.g. 50/100/150/200")
    parser.add_argument("--missing-k", type=int, required=True, help="Blackout or missing span size")
    parser.add_argument("--output", required=True, help="Where to write the JSON config")
    parser.add_argument("--results-dir", required=True, help="Directory for checkpoints and outputs")
    parser.add_argument("--train-iters", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--masking", default="cm", choices=["bm", "rm", "mnr", "cm"])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--model", type=int, default=2,
                        choices=[0, 1, 2],
                        help="Model type: 0=DiffWave, 1=SSSD^SA, 2=SSSD^S4")
    parser.add_argument("--test-dataset", default=None,
                        help="Optional: dataset name to use for test data "
                             "(cross-workload experiment). If omitted, the "
                             "same dataset as --dataset is used for testing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "TraceReconstruction-main" / "Datasets" / args.dataset / f"sequence_length_{args.seq_len}"
    train_path = data_dir / "train.npy"

    # Test data can come from a different dataset (cross-workload / cross-app)
    test_dataset = args.test_dataset if args.test_dataset else args.dataset
    test_data_dir = repo_root / "TraceReconstruction-main" / "Datasets" / test_dataset / f"sequence_length_{args.seq_len}"
    test_path = test_data_dir / "test.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    # Model-specific sub-config key: DiffWave and SSSD^S4 share "wavenet_config";
    # SSSD^SA uses "sashimi_config".
    if args.model == 0:  # DiffWave — no S4 params, needs dilation_cycle
        model_subconfig_key = "wavenet_config"
        model_subconfig = {
            "in_channels": 1,
            "out_channels": 1,
            "num_res_layers": 36,
            "res_channels": 256,
            "skip_channels": 256,
            "dilation_cycle": 10,
            "diffusion_step_embed_dim_in": 128,
            "diffusion_step_embed_dim_mid": 512,
            "diffusion_step_embed_dim_out": 512,
        }
    elif args.model == 2:  # SSSD^S4 — needs S4 params, no dilation_cycle
        model_subconfig_key = "wavenet_config"
        model_subconfig = {
            "in_channels": 1,
            "out_channels": 1,
            "num_res_layers": 36,
            "res_channels": 256,
            "skip_channels": 256,
            "diffusion_step_embed_dim_in": 128,
            "diffusion_step_embed_dim_mid": 512,
            "diffusion_step_embed_dim_out": 512,
            "s4_lmax": args.seq_len,
            "s4_d_state": 64,
            "s4_dropout": 0.0,
            "s4_bidirectional": 1,
            "s4_layernorm": 1,
        }
    else:  # SSSD^SA (use_model=1)
        model_subconfig_key = "sashimi_config"
        model_subconfig = {
            "in_channels": 1,
            "out_channels": 1,
            "d_model": 128,
            "n_layers": 6,
            "diffusion_step_embed_dim_in": 128,
            "diffusion_step_embed_dim_mid": 512,
            "diffusion_step_embed_dim_out": 512,
            "label_embed_dim": 128,
            "label_embed_classes": 71,
            "bidirectional": 1,
            "s4_lmax": args.seq_len,
            "s4_d_state": 64,
            "s4_dropout": 0.0,
            "s4_bidirectional": 1,
        }

    # For cross-workload experiments, encode the test dataset in the dir name
    if args.test_dataset and args.test_dataset != args.dataset:
        run_label = f"{args.dataset}_to_{test_dataset}_seq{args.seq_len}_k{args.missing_k}"
    else:
        run_label = f"{args.dataset}_seq{args.seq_len}_k{args.missing_k}"
    results_dir = Path(args.results_dir).resolve() / run_label
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "diffusion_config": {
            "T": 200,
            "beta_0": 0.0001,
            "beta_T": 0.02,
        },
        model_subconfig_key: model_subconfig,
        "train_config": {
            "output_directory": str(results_dir),
            "ckpt_iter": "max",
            "iters_per_ckpt": args.checkpoint_every,
            "iters_per_logging": args.log_every,
            "n_iters": args.train_iters,
            "learning_rate": args.learning_rate,
            "batch_size_per_gpu": args.batch_size,
            "only_generate_missing": 1,
            "use_model": args.model,
            "masking": args.masking,
            "missing_k": args.missing_k,
        },
        "trainset_config": {
            "train_data_path": str(train_path.resolve()),
            "test_data_path": str(test_path.resolve()),
            "segment_length": args.seq_len,
            "sampling_rate": 1,
            "test_batch_size": 125,
        },
        "gen_config": {
            "output_directory": str(results_dir),
            "ckpt_path": str(results_dir),
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()

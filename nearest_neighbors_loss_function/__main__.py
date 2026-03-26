import argparse
from .utils.train import train_triplet
from .utils.evaluate_model import test_model
from .utils.datasets import get_loaders
import nearest_neighbors_loss_function.utils.set_torch_geometrics
import torch
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Siamese graph neural net training")

    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of random seeds")

    parser.add_argument("--training_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gamma_recalculation_strategy", type=int, required=True)
    parser.add_argument("--gamma_function", type=str, required=True, default=None)
    parser.add_argument("--weight_distances", action="store_false")

    parser.add_argument("--focal_pow", type=float, required=False, default=None)

    parser.add_argument("--density_awareness", action="store_false")
    parser.add_argument("--density_function", type=str, required=False)
    parser.add_argument("--samples_difficultness", action="store_false")
    parser.add_argument("--lambda_samples_difficultness", type=float, default=0.0)

    parser.add_argument("--n_neighbors", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--model_hidden_channels", type=int, required=True)
    parser.add_argument("--model_in_channels", type=int, required=True)
    parser.add_argument("--embedding_length", type=int, required=True)
    parser.add_argument("--optimized_param_name", type=str, required=True)
    parser.add_argument("--debug", action="store_false")

    return parser.parse_args()


def main():
    
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_loaders(args.batch_size, debug=args.debug)

    statistics, best_model_dir_path, n_train_samples = train_triplet(
        seeds=args.seeds,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_type=args.training_type,
        batch_size=args.batch_size,
        gamma_recalculation_strategy=args.gamma_recalculation_strategy,
        gamma_function=args.gamma_function,
        weight_distances=args.weight_distances,
        focal_pow=args.focal_pow,
        density_awareness=args.density_awareness,
        density_function=args.density_function,
        samples_difficultness=args.samples_difficultness,
        lambda_samples_difficultness=args.lambda_samples_difficultness,
        n_neighbors=args.n_neighbors,
        n_epochs=args.n_epochs,
        save_path=args.save_path,
        lr=args.lr,
        model_in_channels=args.model_in_channels,
        model_hidden_channels=args.model_hidden_channels,
        embedding_length=args.embedding_length,
        optimized_param_name=args.optimized_param_name,
        device=device
    )

    statistics = test_model(statistics, best_model_dir_path, args.model_in_channels, args.model_hidden_channels, args.embedding_length, train_loader, 
            n_train_samples, test_loader, args.embedding_length, args.n_neighbors, device)

    statistics.save()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
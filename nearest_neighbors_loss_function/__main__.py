import argparse
from .utils.workflows import train_workflow, test_workflow
import nearest_neighbors_loss_function.utils.set_torch_geometrics
import logging

def parse_workflow_args():
    parser = argparse.ArgumentParser(description="Workflow name parsing")    
    parser.add_argument("--workflow", type=str, required=True)
    args, _ = parser.parse_known_args()
    return args.workflow


def parse_training_args():
    parser = argparse.ArgumentParser(description="Siamese graph neural net training")

    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of random seeds")
    
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    parser.add_argument("--training_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--triplet_loss_margin", type=float, default=0.5)
    parser.add_argument("--batch_shaper_margin", type=float, default=0.5)
    parser.add_argument("--gamma_recalculation_strategy", type=int, required=True)
    parser.add_argument("--gamma_function", type=str, required=True, default=None)
    parser.add_argument("--weight_distances", action="store_true")

    parser.add_argument("--focal_pow", type=float, required=False, default=None)

    parser.add_argument("--density_awareness", action="store_true")
    parser.add_argument("--density_function", type=str, required=False)
    parser.add_argument("--samples_difficultness", action="store_true")
    parser.add_argument("--lambda_samples_difficultness", type=float, default=0.0)

    parser.add_argument("--evaluation_model_name", type=str, default="knn")
    parser.add_argument("--n_evaluation_models", type=int, default=3)

    parser.add_argument("--n_neighbors", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_hidden_channels", type=int, required=True)
    parser.add_argument("--model_in_channels", type=int, required=True)
    parser.add_argument("--model_n_blocks", type=int, required=True)
    parser.add_argument("--embedding_length", type=int, required=True)
    parser.add_argument("--optimized_param_name", type=str, required=True)
    parser.add_argument("--early_stop_window_size", type=int, default=10)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_known_args()[0]
    return args


def parse_testing_args():
    parser = argparse.ArgumentParser(description="Siamese graph neural net testing")

    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    parser.add_argument("--n_evaluation_models", type=int, default=3)
    parser.add_argument("--evaluation_model_name", type=str, default="knn")
    parser.add_argument("--n_neighbors", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_hidden_channels", type=int, required=True)
    parser.add_argument("--model_in_channels", type=int, required=True)
    parser.add_argument("--model_n_blocks", type=int, required=True)
    parser.add_argument("--embedding_length", type=int, required=True)
    parser.add_argument("--optimized_param_name", type=str, required=True)

    args = parser.parse_known_args()[0]
    return args


def main():
    
    workflow = parse_workflow_args()
    
    if workflow == "train_workflow":
        args = parse_training_args()
        train_workflow(args)

    elif workflow == "test_workflow":
        args = parse_testing_args()
        print(args)
        test_workflow(args)

    else:
        raise Exception(f"Workflow {workflow} not implmeneted")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
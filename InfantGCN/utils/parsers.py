import argparse

def get_base_parser():
    """Create the base parser with common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CTRGCN", help="The model used for recognition", type=str)
    parser.add_argument("--data_path", help="The data used for recognition", type=str)
    parser.add_argument("--output_folder", help="Output folder to save the results", type=str)
    parser.add_argument("--exp_name", help="Name of the experiments", type=str)
    return parser

def get_training_parser():
    """Extend the base parser with training-specific arguments."""
    parser = get_base_parser()
    parser.add_argument("--base_lr", default=0.1, help="Base learning rate", type=float)
    parser.add_argument("--epochs", default=20, help="Number of epochs to train the dataset", type=int)
    parser.add_argument("--repeat", default=1, help="Number of times to repeat training dataset", type=int)
    return parser

def get_testing_parser():
    """Extend the base parser with testing-specific arguments."""
    parser = get_base_parser()
    parser.add_argument("--weights", help="Path to the model weights file", type=str)
    return parser

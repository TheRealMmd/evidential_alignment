import argparse


EXPR_PATH = "/path/to/experiments"

dataset_paths = {
    "waterbirds": "/path/to/datasets/waterbirds",
    "celeba": "/path/to/datasets/celeba",
    "multinli": "/path/to/datasets/multinli",
    "civilcomments": "/path/to/datasets/civilcomments",
    "imagenet-9": "/path/to/datasets/imagenet",
    "imagenet-a": "/path/to/datasets/imagenet-a",
    "imagenet-bg": "/path/to/datasets/imagenet-bg",
    "chexpert": "/path/to/datasets/chexpert",
}

NICO_DATA_FOLDER = "/path/to/nico_dataset"
NICO_CXT_DIC_PATH = "/path/to/context_mapping.json"
NICO_CLASS_DIC_PATH = "/path/to/class_mapping.json"


# ============================================================
# Helpers
# ============================================================

def parse_bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Taken from:
# https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value_str = value.split("=")

            if value_str.replace("-", "").isnumeric():
                processed_val = int(value_str)

            elif value_str.replace("-", "").replace(".", "").isnumeric():
                processed_val = float(value_str)

            elif "[" in value_str:
                eles = (
                    value_str
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                )

                processed_val = [
                    int(e.strip())
                    for e in eles
                ]

            elif value_str in ["True", "true"]:
                processed_val = True

            elif value_str in ["False", "false"]:
                processed_val = False

            else:
                processed_val = value_str

            getattr(namespace, self.dest)[key] = processed_val


# ============================================================
# Dataset arguments
# ============================================================

def data_args(parser):

    parser.add_argument(
        "--dataset",
        default="waterbirds",
        type=str,
        help="select dataset",
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )

    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="number of workers",
    )

    parser.add_argument(
        "--resolution",
        default=224,
        type=int,
        help="input image resolution",
    )

    parser.add_argument(
        "--use_shortcutwalk_dataset",
        default=False,
        type=parse_bool,
        help="specify whether to use the shortcutwalk dataset",
    )


# ============================================================
# AFR arguments
# ============================================================

def afr_args(parser):

    parser.add_argument(
        "--balance_classes",
        default=False,
        type=parse_bool,
        help="balance classes when computing the weights",
    )

    parser.add_argument(
        "--group_uniform",
        default=False,
        type=parse_bool,
        help="balance groups when computing the weights",
    )

    parser.add_argument(
        "--afr_gamma",
        default=0,
        type=float,
        help="gamma for computing the weights",
    )

    parser.add_argument(
        "--afr_reg_coeff",
        default=0,
        type=float,
        help="regularization strength for AFR",
    )

    parser.add_argument(
        "--afr_epochs",
        default=20,
        type=int,
        help="number of epochs for AFR last-layer retraining",
    )


# ============================================================
# General training arguments
# ============================================================

def training_args(parser):

    parser.add_argument(
        "--threshold",
        default=0.0,
        type=float,
        help="threshold for selecting spurious dimensions",
    )

    parser.add_argument(
        "--use_relu",
        default="",
        action="store_true",
        help="use relu in the bert model",
    )

    parser.add_argument(
        "--keep_ndims",
        default=-1,
        type=int,
        help="select how many dimensions to keep",
    )

    parser.add_argument(
        "--save_folder",
        default="",
        type=str,
        help="folder that saves experimental results",
    )

    parser.add_argument(
        "--erm_model",
        default="",
        type=str,
        help="ERM trained model",
    )

    parser.add_argument(
        "--train_split",
        default="train",
        type=str,
        help=(
            "Specify which split of the dataset is used for training. "
            "Default is train. When split_train < 1, this can be "
            "train_subset1."
        ),
    )

    parser.add_argument(
        "--test_split",
        default="test",
        nargs="+",
        type=str,
        help=(
            "Specify which split of the dataset is used for testing. "
            "Default is test."
        ),
    )

    parser.add_argument(
        "--mode",
        default="debug",
        type=str,
        help="training mode",
    )

    parser.add_argument(
        "--pretrained",
        default=True,
        type=parse_bool,
        help="load pretrained weights",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "bert_adamw",
        ],
        help="select optimizer",
    )

    parser.add_argument(
        "--optimizer_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr": 1e-3,
        },
    )

    parser.add_argument(
        "--optimizer_backbone",
        type=str,
        default="sgd",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "bert_adamw",
        ],
        help="select backbone optimizer",
    )

    parser.add_argument(
        "--optimizer_backbone_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr": 1e-3,
        },
    )

    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        help="number of epochs to train the main model",
    )

    parser.add_argument(
        "--scheduler",
        default="none",
        type=str,
        help="choose a learning rate scheduler",
    )

    parser.add_argument(
        "--scheduler_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
    )

    parser.add_argument(
        "--scheduler_cls",
        default="none",
        type=str,
        help="choose a classifier learning rate scheduler",
    )

    parser.add_argument(
        "--scheduler_cls_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
    )

    parser.add_argument(
        "--optimizer_cls",
        type=str,
        default="sgd",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "bert_adamw",
        ],
        help="select classifier optimizer",
    )

    parser.add_argument(
        "--optimizer_cls_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr": 1e-4,
        },
    )

    parser.add_argument(
        "--optimizer_vec",
        type=str,
        default="sgd",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "bert_adamw",
        ],
        help="select vector optimizer",
    )

    parser.add_argument(
        "--optimizer_vec_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "lr": 1e-4,
        },
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index",
    )

    parser.add_argument(
        "--check_point",
        type=str,
        default="",
        help="path to a saved model checkpoint",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="frequency of evaluating the model during training",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help=(
            "frequency of saving the model. "
            "If <= 0, intermediate checkpoints are not saved."
        ),
    )

    parser.add_argument(
        "--split_train",
        default=1.0,
        type=float,
        help=(
            "ratio for splitting training data into two parts; "
            "1.0 means no splitting"
        ),
    )

    parser.add_argument(
        "--split_val",
        default=1.0,
        type=float,
        help=(
            "ratio for splitting validation data into two parts; "
            "1.0 means no splitting"
        ),
    )

    parser.add_argument(
        "--num_batches",
        default=200,
        type=int,
        help="number of batches",
    )

    parser.add_argument(
        "--class_balanced",
        default=False,
        type=parse_bool,
        help="use class-balanced sampling during training",
    )

    parser.add_argument(
        "--group_balanced",
        default=False,
        type=parse_bool,
        help="use group-balanced sampling during training",
    )


# ============================================================
# DFR arguments
# ============================================================

def dfr_args(parser):

    parser.add_argument(
        "--dfr_reg",
        default=1.0,
        type=float,
        help="regularization strength for the l1 norm",
    )

    parser.add_argument(
        "--process_embeddings",
        default=False,
        type=parse_bool,
        help="choose whether to normalize embeddings per dimension",
    )

    parser.add_argument(
        "--dfr_epochs",
        default=20,
        type=int,
        help="number of epochs for last-layer retraining",
    )


# ============================================================
# Evidential Alignment arguments
# ============================================================

def evidential_alignment_args(parser):

    parser.add_argument(
        "--edl_epochs",
        type=int,
        default=10,
        help="number of epochs for EDL phase",
    )

    parser.add_argument(
        "--alignment_epochs",
        type=int,
        default=10,
        help="number of epochs for alignment phase",
    )

    parser.add_argument(
        "--edl_lr",
        type=float,
        default=0.01,
        help="learning rate for EDL phase",
    )

    parser.add_argument(
        "--alignment_lr",
        type=float,
        default=0.01,
        help="learning rate for alignment phase",
    )

    parser.add_argument(
        "--edl_weight_decay",
        type=float,
        default=0.0,
        help="weight decay for EDL phase",
    )

    parser.add_argument(
        "--alignment_weight_decay",
        type=float,
        default=0.0,
        help="weight decay for alignment phase",
    )

    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.1,
        help="weight for regularization term",
    )

    parser.add_argument(
        "--annealing_step",
        type=int,
        default=10,
        help="step size for EDL annealing",
    )

    parser.add_argument(
        "--kl_reg",
        type=float,
        default=1.0,
        help="regularization strength for KL distance",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for Evidential Alignment weights",
    )


# ============================================================
# RATER arguments
# ============================================================

def rater_args(parser):
    """
    Hyperparameters for the feature-level bilevel Rater.

    The Rater operates on frozen backbone embeddings:

        image
          -> frozen ImageNet-pretrained ResNet-50
          -> embedding z
          -> rater(z)
          -> scalar score

    The score is converted to a softmax weight and used in the
    differentiable inner optimization.
    """

    parser.add_argument(
        "--rater_meta_steps",
        type=int,
        default=100,
        help="number of outer/meta optimization steps",
    )

    parser.add_argument(
        "--rater_inner_steps",
        type=int,
        default=2,
        help=(
            "number of differentiable inner classifier updates "
            "performed per meta step"
        ),
    )

    parser.add_argument(
        "--rater_num_inner_models",
        type=int,
        default=4,
        help="number of inner classifier models in the population",
    )

    parser.add_argument(
        "--rater_inner_lr",
        type=float,
        default=1e-2,
        help="learning rate for differentiable inner updates",
    )

    parser.add_argument(
        "--rater_outer_lr",
        type=float,
        default=3e-4,
        help="learning rate for updating the embedding rater",
    )

    parser.add_argument(
        "--rater_temperature",
        type=float,
        default=2.0,
        help=(
            "temperature used in softmax(rating / temperature) "
            "to convert ratings to inner-loop weights"
        ),
    )

    parser.add_argument(
        "--rater_refresh_steps",
        type=int,
        default=100,
        help=(
            "number of meta steps defining the inner-model "
            "population refresh period"
        ),
    )

    parser.add_argument(
        "--rater_grad_clip",
        type=float,
        default=5.0,
        help="gradient clipping norm for the rater",
    )

    parser.add_argument(
        "--rater_capacity",
        type=str,
        default="medium",
        choices=[
            "small",
            "medium",
        ],
        help="architecture/capacity of the embedding rater",
    )

    parser.add_argument(
        "--rater_outer_reg_weight",
        type=float,
        default=0.0,
        help="L2 regularization weight on the rater parameters",
    )

    parser.add_argument(
        "--rater_inner_reg_weight",
        type=float,
        default=0.0,
        help=(
            "L2 regularization weight on the differentiable "
            "inner classifier parameters"
        ),
    )

    parser.add_argument(
        "--rater_score_reg_weight",
        type=float,
        default=0.0,
        help=(
            "weight of the score-diversity regularizer; "
            "0 disables it"
        ),
    )


# ============================================================
# Main parser
# ============================================================

def get_args():

    parser = argparse.ArgumentParser(
        description="spurious correlation"
    )

    # Dataset/general algorithms
    data_args(parser)
    training_args(parser)

    # Algorithm-specific arguments
    dfr_args(parser)
    afr_args(parser)
    evidential_alignment_args(parser)
    rater_args(parser)

    parser.add_argument(
        "--algorithm",
        default="erm",
        type=str,
        help="training algorithm",
    )

    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="choose the backbone network",
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="load a saved model",
    )

    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="additional information",
    )

    parser.add_argument(
        "--test_erm",
        default="",
        type=str,
        help="additional information",
    )

    parser.add_argument(
        "--use_val",
        default="",
        action="store_true",
        help="use validation set for training",
    )

    parser.add_argument(
        "--add_train",
        default="",
        action="store_true",
        help="use validation and training sets for training",
    )

    parser.add_argument(
        "--no_mask",
        default="",
        action="store_true",
        help="disable masked linear layer",
    )

    parser.add_argument(
        "--identify_split",
        default="",
        type=str,
        help="specify the split of identification data",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--jtt_lambda",
        type=float,
        default=100,
    )

    parser.add_argument(
        "--first_stage_epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--result_path",
        default="",
        type=str,
        help="specify where to store evaluation results",
    )

    args = parser.parse_args()

    return args

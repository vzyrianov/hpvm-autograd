from torch import Tensor


def accuracy(output: Tensor, target: Tensor) -> float:
    """The "classification accuracy" metric (return value is between 0 and 100).
    
    :param output: Probability distribution output from the model.
    :param target: A 1d-tensor of labels, one for each input image, from the dataset.
    """

    _, pred_labels = output.max(1)
    n_correct = (pred_labels == target).sum().item()
    return n_correct / len(output) * 100

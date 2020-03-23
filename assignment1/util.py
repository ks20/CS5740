from propername import propername_data_loader
from newsgroup import newsgroup_data_loader
from pandas import DataFrame 

def save_results(predictions, results_path):
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    # TODO: Implement saving of the results.
    results = DataFrame({'id': range(len(predictions)), 'newsgroup': predictions})
    results.to_csv(results_path, index = False)
    

def compute_accuracy(labels, predictions):
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
        idx_inc: list of IDs for incorrect classifications
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as " \
                         "length of predictions (" + str(len(predictions)) + ".")
    # TODO: Implement accuracy computation.
    idx_inc = []
    numCorrect = 0
    for i in range(len(labels)):
        if (labels[i] == predictions[i]):
            numCorrect += 1
        else:
            idx_inc.append(i)

    return numCorrect/len(labels), idx_inc

def evaluate(model, data, results_path):
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    
    Returns:
        float representing the % of predictions that were true.
        idx_inc: list of IDs for incorrect classifications
    """

    predictions = [model.predict(example[0]) for example in data]
    labels = [example[1] for example in data]
    # print('\n\n\n')
    # print(predictions)
    # print('\n\n\n')
    # print(labels)

    # print([example[0] for example in data])
    # print(predictions)
    save_results(predictions, results_path)
    results = compute_accuracy(labels, predictions)

    return results[0], results[1]

def load_data(args):
    """ Loads the data.

    Inputs:
        args (list of str): The command line arguments passed into the script.

    Returns:
        Training, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    data_type = ""
    if 'propername' in args:
      data_loader = propername_data_loader
      data_type = "propername"
    elif 'newsgroup' in args:
      data_loader = newsgroup_data_loader
      data_type = "newsgroup"
    assert data_loader, "Choose between newsgroup or propername data. " \
                        + "Args was: " + str(args)

    # Load the data. 
    train_data, dev_data, test_data = data_loader("data/" + data_type +  "s/train/train_data.csv",
                                                  "data/" + data_type + "s/train/train_labels.csv",
                                                  "data/" + data_type + "s/dev/dev_data.csv",
                                                  "data/" + data_type + "s/dev/dev_labels.csv",
                                                  "data/" + data_type + "s/test/test_data.csv")

    return train_data, dev_data, test_data, data_type

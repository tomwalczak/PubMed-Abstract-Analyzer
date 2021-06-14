import pandas as pd
import numpy as np

import tensorflow as tf
import zipfile
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# nltk.download('punkt')
# from nltk import tokenize

def get_abstract_results_df(model,sent_detector,class_names,full_abstract):
 
    sentences = sent_detector.tokenize(full_abstract)

    sentences = clean_sents_not_starting_with_uppercase(sentences)

    preds = model.predict(add_positon_feature_to_sentences(sentences))

    return get_model_preds_as_df(None,preds,sentences,class_names)

def add_positon_feature_to_sentences(sentences):
  return ["POSITION_" + (np.around(line_num / len(sentences),decimals=2)*100).astype("int").astype("str") + " " + sentence for line_num, sentence in enumerate(sentences)]


def get_model_preds_as_df(y_true_labels_int, y_preds, sentences, class_names,):

  # Are y_preds with probs for each class or just predicted class (1-dim)?
  if y_preds.ndim == 2:

      pred_classes = y_preds.argmax(axis=1)
      pred_conf = y_preds.max(axis=1)

      # If we have true labels (valid set)
      if y_true_labels_int:

          pred_df = pd.DataFrame({
              "y_true": y_true_labels_int,
              "y_pred": pred_classes,
              "y_true_class_name": [class_names[pred] for pred in y_true_labels_int],
              "y_pred_class_name": [class_names[pred] for pred in pred_classes],
              "confidence": [ '{cnf}%'.format(cnf=int(conf*100)) for conf in pred_conf],
          })
          pred_df["is_pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
      else: # If we don't, i.e. inference
          pred_df = pd.DataFrame({
              "y_pred": pred_classes,
              "y_pred_class_name": [class_names[pred] for pred in pred_classes],
              "confidence": [ '{cnf}%'.format(cnf=int(conf*100)) for conf in pred_conf],
          })

      pred_df["sentence"] = sentences

      return pred_df
  else:

    pred_df = pd.DataFrame({
    'sentence': sentences,
    'y_pred_class_name': class_names[y_preds]
    })

    return pred_df



def clean_sents_not_starting_with_uppercase(sentences):
    proc_sents = []
    prev_sent = ""
    for line_num, new_sent in enumerate(sentences):
        # if sent starts with uppercase, 
        if new_sent[0].isupper():
            prev_sent and proc_sents.append(prev_sent) 
            prev_sent = new_sent
        else:
         prev_sent = prev_sent + new_sent

    return proc_sents


def preprocess_abstracts_from_file(filename):
  """
  Returns a list of abstracts.

  """
  abstracts = []

  input_lines = get_lines(filename) # get all lines from filename
  abstract_lines = "" # create an empty abstract
  previous_class = "" # extra feautre for context
  abstract_samples = [] # create an empty list of abstracts
  

  # Loop through each line in the target file
  for line in input_lines:
    if line.startswith("###"): # check to see if the line is an ID line
      abstract_lines and abstracts.append(abstract_lines) # append previous abstract as it's finished now
      abstract_id = line
      abstract_lines = "" # reset the abstract string if the line is an ID line
    else:
      abstract_lines = abstract_lines + line

  # Append the remainig abstract
  abstracts.append(abstract_lines)

  return abstracts

def get_abstract_markdown(abstract_bd_df):
  last_class = ""
  markdown = ""
  for idx, line in abstract_bd_df.iterrows():
    if line['y_pred_class_name'] != last_class:
      last_class = line['y_pred_class_name']
      markdown = '{markdown}  \n  ####  {last_class} \n {sentence}   '.format(markdown=markdown, last_class=last_class, sentence=line['sentence'])
    else:
      markdown = '{markdown} {sentence} '.format(markdown=markdown, sentence=line['sentence'])

  return markdown

def get_lines(filename):
  """
  Reads filename and returns the lines of text as a list.


  """
  with open(filename, "r") as f:
    return f.readlines()

# Credit: @mrdbourke
def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

# Remix of Scikit-Learn's@mrdbourke's implementation


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if len(classes) > 1:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")


def model_preds_df(y_true_labels_int, y_preds, preds_probs, sentences, class_names):
  pred_classes = preds_probs.argmax(axis=1)

  pred_df = pd.DataFrame({
      "y_true": y_true_labels_int,
      "y_pred": y_preds,
      "y_true_class_name": [class_names[pred] for pred in y_true_labels_int],
      "y_pred_class_name": [class_names[pred] for pred in y_preds],
      "pred_confidence": preds_probs.max(axis=1),
  })

  pred_df["is_pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
  pred_df["sentence"] = sentences

  return pred_df

def get_wrong_preds(preds_df):
  wrong_preds_df = preds_df.loc[preds_df['is_pred_correct'] == False]
  most_wrong_preds = wrong_preds_df.sort_values('pred_confidence',ascending=False)[["pred_confidence","y_true_class_name","y_pred_class_name", "sentence"]]
  sample_wrong_preds = most_wrong_preds.sample(100)

  return wrong_preds_df, most_wrong_preds, sample_wrong_preds.sort_values('pred_confidence',ascending=False)

# Credit: @mrdbourke
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Plot the validation and training data separately

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Credit: @mrdbourke
def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
    return model_results


def preprocess_text_add_line_position_features(filename):
  """
  Returns a list of dictionaries of abstract line data, with position features
  Returns a list of full abstracts without target lables

  Takes in filename, reads it contents and sorts through each line,
  extracting things like the target label, the text of the sentnece,
  how many sentences are in the current abstract and what sentence
  number the target line is.
  """
  input_lines = get_lines(filename) # get all lines from filename
  abstract_lines = "" # create an empty abstract
  abstract_samples = [] # create an empty list of abstracts
  full_abstracts = []
  current_abstract = ""

  # Loop through each line in the target file
  for line in input_lines:
      if line.startswith("###"): # check to see if the line is an ID line

          current_abstract and full_abstracts.append(current_abstract)
          abstract_lines = "" # reset the abstract string if the line is an ID line
          current_abstract = ""

      elif line.isspace(): # check to see if line is a new line
          abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

          # Iterate through each line in a single abstract and count them at the same time
          for abstract_line_number, abstract_line in enumerate(abstract_line_split):
              line_data = {} # create an empty dictionary for each line
              target_text_split = abstract_line.split("\t") # split target label from text 
              line_data["target"] = target_text_split[0] # get target label
              line_data["text"] = target_text_split[1].lower() # get target text and lower it
              current_abstract += line_data["text"]
              line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
              line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are there in the target abstract? (start from 0)
              line_data["text_with_pos_feature"] = "POSITION_" + (np.around(line_data["line_number"] / line_data["total_lines"], decimals=2)*100).astype("int").astype("str") + " " + line_data["text"]

                  
              abstract_samples.append(line_data) # add line data to abstract samples list

      else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
          abstract_lines += line

  return abstract_samples, full_abstracts
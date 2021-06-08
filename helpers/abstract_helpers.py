import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk import tokenize

def get_abstract_results_df_nb(nb_model,class_names,full_abstract):
    sentences = tokenize.sent_tokenize(full_abstract)

    sentences = clean_sents_not_starting_with_uppercase(sentences)

    nb_preds = nb_model.predict(add_positon_feature_to_sentences(sentences))

    preds_df = pd.DataFrame({
    'sentence': sentences,
    'y_pred_class_name': class_names[nb_preds]
    })

    return preds_df


def get_abstract_results_df(model,class_names,full_abstract):
  sentences = tokenize.sent_tokenize(full_abstract)

  sentences = clean_sents_not_starting_with_uppercase(sentences)

  pred = model.predict(add_positon_feature_to_sentences(sentences))

  return get_model_preds_as_df(None,pred,sentences,class_names)

def add_positon_feature_to_sentences(sentences):
  return ["POSITION_" + (np.around(line_num / len(sentences),decimals=2)*100).astype("int").astype("str") + " " + sentence for line_num, sentence in enumerate(sentences)]


def get_model_preds_as_df(y_true_labels_int, y_preds, sentences, class_names):
  
  pred_classes = y_preds.argmax(axis=1)
  pred_conf = y_preds.max(axis=1)

  # Cover the use case for inference rather than test set
  if y_true_labels_int:

      pred_df = pd.DataFrame({
        "y_true": y_true_labels_int,
        "y_pred": pred_classes,
        "y_true_class_name": [class_names[pred] for pred in y_true_labels_int],
        "y_pred_class_name": [class_names[pred] for pred in pred_classes],
        "confidence": [ '{cnf}%'.format(cnf=int(conf*100)) for conf in pred_conf],
       })
      pred_df["is_pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
  else:
    pred_df = pd.DataFrame({
        "y_pred": pred_classes,
        "y_pred_class_name": [class_names[pred] for pred in pred_classes],
        "confidence": [ '{cnf}%'.format(cnf=int(conf*100)) for conf in pred_conf],
    })

  pred_df["sentence"] = sentences

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


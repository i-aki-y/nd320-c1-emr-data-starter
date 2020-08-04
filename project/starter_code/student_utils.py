import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    ndc_df = ndc_df[["NDC_Code", "Non-proprietary Name"]]
    ndc_df.columns = ["ndc_code", "generic_drug_name"]
    df = df.merge(ndc_df, on="ndc_code", how="left")
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''

    first_encounter_df = df.groupby(["patient_nbr"]).first().reset_index()
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    keys = df[patient_key].values
    np.random.seed(seed=13) # for reproduciblity
    keys = np.random.permutation(keys)
    
    n = len(keys)
    n_train = int(0.6*n)
    n_validation = int(0.2*n)

    keys_train = keys[:n_train]
    keys_validation = keys[n_train:n_train+n_validation]
    keys_test = keys[n_train+n_validation:]

    train = df[df[patient_key].isin(keys_train)]
    validation = df[df[patient_key].isin(keys_validation)]
    test = df[df[patient_key].isin(keys_test)]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(
            categorical_column
        )
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''

    col = tf.cast(col, tf.float32)
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(col, default_value=default_value, 
                                                          normalizer_fn=lambda x: normalize_numeric_with_zscore(x, MEAN, STD))
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    threshold = 5 # include the data if the prediction is more than 3 days
    student_binary_prediction = np.where(df[col] > threshold, 1, 0)
    return student_binary_prediction

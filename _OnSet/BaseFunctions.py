import pickle
import sys
from datetime import datetime as dt
import numpy as np
import os
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier

now = dt.now()
runtime = now.strftime('%d-%m-%Y_%H%M')

INTERVENTION = 'vent'
RANDOM = 34000
MAX_LEN = 240
SLICE_SIZE = 6
PREDICTION_WINDOW = 4
OUTCOME_TYPE = 'all'
NUM_CLASSES = 2

# %%

CHUNK_KEY = {'ONSET': 0, 'CONTROL': 1}


# %% md

# Load Data

# %%
def RemoveSpecialChars(str):
    str = str.replace('/', ' ')
    str = str.replace(' ', '_')
    alphanumeric = ''
    for character in str:
        if character.isalnum() or character == '_':
            alphanumeric += character
    return alphanumeric


DATAFILE = 'all_hourly_data_3000.h5'
C_FILE = 'C.h5'
ARDS_HF ='static_ards_hf.csv'


def mksureDir(outdir):
    try:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
    except:
        pass


mksureDir('./Model_Results')

UltimoPath = "Model_Results/" + runtime + '__' + RemoveSpecialChars(DATAFILE)[-12:] + RemoveSpecialChars(__file__)


def loadModel(pathToModel):
    try:
        with open(pathToModel, mode='rb') as f:Pred_Model = pickle.load(f)
        return Pred_Model
    except:
        return ''



def minmax(x):  # normalize
    mins = x.min()
    maxes = x.max()
    x_std = (x - mins) / (maxes - mins)
    return x_std


# %%

def std_time_since_measurement(x):
    idx = pd.IndexSlice
    x = np.where(x == 100, 0, x)
    means = x.mean()
    stds = x.std()
    x_std = (x - means) / stds
    return x_std


## Categorization of Static Features

# %%

def categorize_age(age):
    if age > 10 and age <= 30:
        cat = 1
    elif age > 30 and age <= 50:
        cat = 2
    elif age > 50 and age <= 70:
        cat = 3
    else:
        cat = 4
    return cat


def categorize_ethnicity(ethnicity):
    if 'AMERICAN INDIAN' in ethnicity:
        ethnicity = 'AMERICAN INDIAN'
    elif 'ASIAN' in ethnicity:
        ethnicity = 'ASIAN'
    elif 'WHITE' in ethnicity:
        ethnicity = 'WHITE'
    elif 'HISPANIC' in ethnicity:
        ethnicity = 'HISPANIC/LATINO'
    elif 'BLACK' in ethnicity:
        ethnicity = 'BLACK'
    else:
        ethnicity = 'OTHER'
    return ethnicity


def find_opt_dt(tree_model, data2, max_depth=30):
    """
    Find the optimal depth which generalizes well on the test set for a simple decision tree
    :param tree_model: (sklearn.tree.DecisionTreeRegressor object)
    :param data2: (tuple) 4 element tuple containing a shuffled X_train, X_test, y_train, y_test splits of the data
    :param max_depth: (scalar) optimal value for decision tree depth which gives the best metric on test set
    :return: (tuple) (sklearn model of optimal depth trained tree, (scalar) optimal depth,
                    (scalar) MSE of trained model on test set)
    """
    _, _, _, y_test = data2
    optimal_depth = 0
    min_mse = np.max(y_test) ** 2
    best_tree = None

    for i in range(max_depth + 1):
        tree = tree_model(max_depth=i + 1)
        trained_tree, mse = fit_and_predict(tree, data2)
        if mse < min_mse:
            min_mse = mse
            optimal_depth = i
            best_tree = trained_tree

    return best_tree, optimal_depth, min_mse


def fit_and_predict(sklearn_model, test_train_data):
    """
    A function for automating the process of fitting and evaluating sklearn models
    :param sklearn_model: (Type: sklearn model object)
    :param test_train_data: (tuple) 4 element tuple containing a shuffled X_train, X_test, y_train, y_test splits of the data
    :return: (tuple) (train sklearn model object, (scalar) model's MSE on the test set)
    """
    X_train2, X_test2, y_train2, y_test2 = test_train_data
    trained_model = sklearn_model.fit( np.nan_to_num(X_train2) , np.nan_to_num(y_train2))
    prediction = trained_model.predict(X_test2)
    mse = mean_squared_error(y_test2, prediction)
    return trained_model, mse


# %%

## Make Tensors

# %%

def create_x_matrix(x):
    zeros = np.zeros((MAX_LEN, x.shape[1] - 4))
    x = x.values
    x = x[:(MAX_LEN), 4:]
    zeros[0:x.shape[0], :] = x
    return zeros


def create_y_matrix(y):
    zeros = np.zeros((MAX_LEN, y.shape[1] - 4))
    y = y.values
    y = y[:, 4:]
    y = y[:MAX_LEN, :]
    zeros[:y.shape[0], :] = y
    return zeros


## Make Windows

# %%

def make_3d_tensor_slices(GAP_TIME,X_tensor, Y_tensor, lengths , staticCol ):
    num_patients = X_tensor.shape[0]
    timesteps = X_tensor.shape[1]
    num_features = X_tensor.shape[2]
    X_tensor_new = np.zeros((lengths.sum(), SLICE_SIZE, num_features + 1))
    Y_tensor_new = np.zeros((lengths.sum()))

    current_row = 0

    for patient_index in range(num_patients):
        x_patient = X_tensor[patient_index]
        y_patient = Y_tensor[patient_index]
        length = lengths[patient_index]

        for timestep in range(length - PREDICTION_WINDOW - GAP_TIME - SLICE_SIZE):
            x_window = x_patient[timestep:timestep + SLICE_SIZE]

            # Remember To RELOCATE <Current supervised Y > on Features table and not on STATIC table !!!!  !!! Roy
            y_window = y_patient[timestep:timestep + SLICE_SIZE]
            ###############ROY ONSET ONLY
            if max(y_window) > 0:
                continue
            ##############
            x_window = np.concatenate((x_window[0:,0:staticCol-1], np.expand_dims(y_window, 1),x_window[0:,staticCol-1:]), axis=1)
            result_window = y_patient[
                            timestep + SLICE_SIZE + GAP_TIME:timestep + SLICE_SIZE + GAP_TIME + PREDICTION_WINDOW]
            result_window_diff = set(np.diff(result_window))
            # if 1 in result_window_diff: pdb.set_trace()
            gap_window = y_patient[timestep + SLICE_SIZE:timestep + SLICE_SIZE + GAP_TIME]
            gap_window_diff = set(np.diff(gap_window))

            ###############ROY ONSET ONLY
            #if max(gap_window) > 0:
                #continue
            ##############


            # print result_window, result_window_diff

            if OUTCOME_TYPE == 'binary':
                if max(gap_window) == 1:
                    result = None
                elif max(result_window) == 1:
                    result = 1
                elif max(result_window) == 0:
                    result = 0
                if result != None:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1

            else:
                if 1 in gap_window_diff or -1 in gap_window_diff:
                    result = None
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 0):
                    result = CHUNK_KEY['CONTROL']
                elif (len(result_window_diff) == 1) and (0 in result_window_diff) and (max(result_window) == 1):
                    result =  CHUNK_KEY['ONSET']
                elif 1 in result_window_diff:
                    result = CHUNK_KEY['ONSET']
                elif -1 in result_window_diff:
                    result = None
                else:
                    result = None

                if result != None:
                    X_tensor_new[current_row] = x_window
                    Y_tensor_new[current_row] = result
                    current_row += 1

    X_tensor_new = X_tensor_new[:current_row, :, :]
    Y_tensor_new = Y_tensor_new[:current_row]

    # Remember To RELOCATE <Current supervised Y > LAST INDEX ON X on Features table and not on STATIC table !!!!  !!! Roy
    return X_tensor_new, Y_tensor_new


def remove_duplicate_static(x,time_series_col):
    x_static = x[:, 0, time_series_col:x.shape[2] ]
    x_timeseries = np.reshape(x[:, :, :time_series_col], (x.shape[0], -1))
    x_int = x[:, :, time_series_col -1]
    x_concat = np.concatenate((x_static, x_timeseries, x_int), axis=1)
    return x_concat


# %% md

## Hyperparameter Generation

# %%

class DictDist():
    def __init__(self, dict_of_rvs): self.dict_of_rvs = dict_of_rvs

    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out


class Choice():
    def __init__(self, options): self.options = options

    def rvs(self, n): return [self.options[i] for i in ss.randint(0, len(self.options)).rvs(n)]


# %%


## Fit model

# %%

def run_basic(GAP_TIME ,model_name, model, hyperparams_list, X_train, X_val, X_test, y_train, y_val_classes, y_test_classes,
              pathToLoadModel=''):

    best_s, best_hyperparams = -np.Inf, None
    loadm=loadModel(pathToLoadModel)
    if  loadm != '':
        best_m = loadm
        best_hyperparams = best_m
    else:
        for i, hyperparams in enumerate(hyperparams_list):
            print(hyperparams);
            # for i_mdp in range(3,4):
            #     for i_n_estimators in range(100,300,50):
            M = model(**hyperparams)
            #M.max_depth=i_mdp;
            #M.n_estimators = i_n_estimators;
            #hyperparams['max_depth']=i_mdp;
            #hyperparams['n_estimators'] = i_n_estimators;
            print("On sample %d / %d (hyperparams = %s)" % (i + 1, len(hyperparams_list), repr((hyperparams))))
            ## ROY M.fit( X_train, y_train)
            M.fit(np.nan_to_num(X_train), y_train)#, eval_set=[(np.nan_to_num(X_train), y_train), (np.nan_to_num(X_val), y_val)])  # ROY  use nan_to_num and ,eval_metric='auc',verbose=True
            if model_name == 'XGB':
                t_values = pd.DataFrame(X_train).columns.values
                u_feature_importance = M.feature_importances_

                # Define a dictionary containing Students data
                feature_importance_data = {'feature_idx': t_values, 'Importance': u_feature_importance}

                # Convert the dictionary into DataFrame
                u_feature_importance_df = pd.DataFrame(feature_importance_data)
                u_feature_importance_df = u_feature_importance_df.sort_values(by='Importance', ascending=False)
                print('feature_importance', u_feature_importance_df)

            valprd = M.predict_proba(np.nan_to_num(X_val))
            roc_auc_score_s = roc_auc_score(y_val_classes, valprd, average='macro')
            print ('MSE -- > {0} \n'.format(roc_auc_score_s))
            if roc_auc_score_s > best_s:
                best_s, best_hyperparams, best_m = roc_auc_score_s, hyperparams, M
                print("New Vvalidation Best Score: %.2f @ hyperparams = %s" % (100 * best_s, repr((best_hyperparams))))



    return roc_auc_score_s, best_m, best_hyperparams,u_feature_importance_df, run_only_final(model_name, best_m, np.nan_to_num(X_test), y_test_classes,GAP_TIME)


def run_only_final(model_name, model, X_test, Y_test , GAP_TIME):
    y_pred = pd.DataFrame(model.predict_proba(X_test))
    Y_test = pd.DataFrame(Y_test)

    mksureDir('./Aucroc')


    # SAVE results
    predcsv = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_pred.csv'
    testcsv = 'Aucroc/Gap '+ str(GAP_TIME) + runtime + RemoveSpecialChars(DATAFILE)[-12:] + '__' + model_name + '_Y_test.csv'
    (y_pred).to_csv(predcsv,
                    index=False)
    (Y_test).to_csv(testcsv,
                    index=False)

    return y_pred ,testcsv,predcsv


# %%

def aucroc(Y_test, y_pred):
    try:

        auc = roc_auc_score(Y_test, y_pred, average=None)
        aucmacro = roc_auc_score(Y_test, y_pred, average='macro')
        #roc_curves= roc_curve(Y_test, y_pred)
        print("auc , aucmacro ", auc, aucmacro)
        #print("roc_curve ",  roc_curves)
    except:
        print("roc_auc_score error ")
        auc=0
        aucmacro=0


    recall = recall_score(Y_test, y_pred.round(), average=None)
    precision = precision_score(Y_test, y_pred.round(), average=None, zero_division=False)
    return auc, aucmacro, recall, precision


def RF_LR_XGB(GAP_TIME,x_train_concat, x_val_concat, x_test_concat, y_train, y_val_classes,  y_test_classes,
              xgb_mod_path,rf_mod_path,lr_mod_path):
    # %% md

    # T = x_train_concat,y_train,x_test_concat,y_test_classes
    # from sklearn.tree import DecisionTreeRegressor
    # decision_tree, opt_depth, dt_min_mse = find_opt_dt(DecisionTreeRegressor, T )

    N_EPOCHES = 1
    #RANDOM=0;
    np.random.seed(RANDOM)

    ##### ROY XGB
    XGB_dist = DictDist({
        'verbosity': Choice([1]),
        'eval_metric': Choice(['auc']),
        # 'early_stopping_rounds': Choice([4]),
        'learning_rate': Choice([0.1]),
        'n_estimators': Choice([125]),
        'max_depth': Choice([3]),  # find_opt_dt ROY
        'random_state': Choice([RANDOM])
        # ,
        # 'gpu_id' :Choice([0]),
        # 'tree_method' : Choice(['gpu_hist'])
    })
    XGB_hyperparams_list = XGB_dist.rvs(N_EPOCHES)

    LR_dist = DictDist({
        'C': Choice(np.geomspace(1e-3, 1e3, 10000)),
        'penalty': Choice(['l2']),
        'solver': Choice(['sag']),
        'max_iter': Choice([100, 200]),
        'class_weight': Choice(['balanced']),
        'multi_class': Choice(['multinomial']),
        'random_state': Choice([RANDOM])
    })
    LR_hyperparams_list = LR_dist.rvs(N_EPOCHES)

    RF_dist = DictDist({
        'n_estimators': ss.randint(50, 200),
        'max_depth': ss.randint(2, 10),
        'min_samples_split': ss.randint(2, 75),
        'min_samples_leaf': ss.randint(1, 50),
        'class_weight': Choice(['balanced']),
        'random_state': Choice([RANDOM])

    })
    RF_hyperparams_list = RF_dist.rvs(N_EPOCHES)

    ########################################### LR + RF + XGB ############################################
    results = {}

    from colorama import Fore, Back, Style
    import colorama
    colorama.init(autoreset=False)

    # i.e 'Saved_Models/06-04-2020_1347___data_3000h5__XGB_finalized'
    xgb_mod_path = xgb_mod_path
    rf_mod_path = rf_mod_path
    lr_mod_path = lr_mod_path

    for model_name, path_to_model, model, hyperparams_list in [
        ('XGB', xgb_mod_path, XGBClassifier, XGB_hyperparams_list)
        # ,
        # ('RF', rf_mod_path, RandomForestClassifier, RF_hyperparams_list),
        # ('LR', lr_mod_path, LogisticRegression, LR_hyperparams_list)
    ]:
        curUltimoPath = UltimoPath + "_" + model_name
        if model_name not in results: results[model_name] = {}

        print(Fore.BLACK + Back.GREEN + "Running model %s " % (model_name))

        #######################################################
        ######### run_basic
        ###################

        val_roc_auc_score_s, best_model, results[model_name], u_feature_importance_df,resb = run_basic(GAP_TIME,
            model_name, model, hyperparams_list, x_train_concat, x_val_concat, x_test_concat, y_train, y_val_classes,
            y_test_classes , path_to_model)

        y_pred = resb[0]
        testcsv = resb[1],
        predcsv = resb[2]

        ##################
        ######### run_basic
        #######################################################

        auc, aucmacro, recall, precision = aucroc(y_test_classes, y_pred )

        print(' auc :{0}, aucmacro:{1} ,recall {2},precision {3} '.format(auc, aucmacro, recall, precision))

        res = {'SET best_hyperparams': results[model_name],'\nVal_auc':val_roc_auc_score_s, '\nTest_auc': auc, '\nTest_aucmacro': aucmacro, '\nTest_recall': recall,
               '\nTest_precision': precision}

        print("Final results for model %s " % (model_name))

        print( 'Gap time '+ str(GAP_TIME) +'h Summary : {0} \n\n'.format(res))
        print(Style.RESET_ALL)

        # Save hyper params and results
        with open(curUltimoPath, mode='wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)



        mksureDir('./Saved_Models')

        # save the MODEL to disk
        filename = 'Saved_Models/GAP'+ str(GAP_TIME)+'_' + runtime + '__' + RemoveSpecialChars(DATAFILE)[
                                                      -12:] + '__' + model_name + '_finalized'
        #pickle.dump(best_model, open(filename, 'wb'))
        with open(filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(best_model, f, pickle.HIGHEST_PROTOCOL)

        print(Fore.RED + Back.WHITE + "DONE %s ! \n" % (model_name))

    # with open(curUltimoPath, mode='rb') as f:resultsx = pickle.load(f)
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.WHITE + "Done!")
    return u_feature_importance_df ,testcsv,predcsv

    # print (results)
    # print(results[model_name])

    # %% md


def GoCNN(class_weight, x_train, y_train, y_train_classes, x_val, y_val_classes, x_test, y_test_classes):
    # CNN

    # %%

    import keras
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Input, Conv1D, MaxPooling1D
    from keras.callbacks import EarlyStopping

    # %%

    # from tensorflow import set_random_seed
    #
    # set_random_seed(RANDOM)

    # %%

    BATCH_SIZE = 128
    EPOCHS = 12
    DROPOUT = 0.5

    # %%

    # %%

    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)
    model = Conv1D(64, kernel_size=3,
                   strides=1,
                   activation='relu',
                   input_shape=input_shape,
                   padding='same',
                   name='conv2')(inputs)

    model = (MaxPooling1D(pool_size=3, strides=1))(model)

    model2 = Conv1D(64, kernel_size=4,
                    strides=1,
                    activation='relu',
                    input_shape=input_shape,
                    padding='same',
                    name='conv3')(inputs)

    model2 = MaxPooling1D(pool_size=3, strides=1)(model2)

    model3 = Conv1D(64, kernel_size=5,
                    strides=1,
                    activation='relu',
                    input_shape=input_shape,
                    padding='same',
                    name='conv4')(inputs)

    model3 = MaxPooling1D(pool_size=3, strides=1)(model3)

    models = [model, model2, model3]

    full_model = keras.layers.concatenate(models)
    full_model = Flatten()(full_model)
    full_model = Dense(128, activation='relu')(full_model)
    full_model = Dropout(DROPOUT)(full_model)
    full_model = Dense(NUM_CLASSES, activation='softmax')(full_model)

    full_model = keras.models.Model(input=inputs, outputs=full_model)

    full_model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(lr=.0005),
                       metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    full_model.fit(x_train, y_train_classes,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   verbose=1,
                   class_weight=class_weight,
                   callbacks=[early_stopping],
                   validation_data=(x_val, y_val_classes))

    # %%

    test_preds_cnn = full_model.predict(x_test, batch_size=BATCH_SIZE)

    # ROY
    print(roc_auc_score(y_test_classes, test_preds_cnn, average=None))
    print(roc_auc_score(y_test_classes, test_preds_cnn, average='macro'))
    print(roc_auc_score(y_test_classes, test_preds_cnn, average='micro'))

    # %% md


def GoLSTM(class_weight, x_train, x_test, x_val, y_train, y_val, y_test, y_val_classes, y_test_classes):
    import tensorflow as tf
    import functools
    BATCH_SIZE = 128
    EPOCHS = 12
    KEEP_PROB = 0.8
    REGULARIZATION = 0.001
    NUM_HIDDEN = [512, 512]

    def lazy_property(function):
        attribute = '_' + function.__name__

        @property
        @functools.wraps(function)
        def wrapper(self):
            if not hasattr(self, attribute):
                setattr(self, attribute, function(self))
            return getattr(self, attribute)

        return wrapper

    class VariableSequenceLabelling:

        def __init__(self, data, target, dropout_prob, reg, num_hidden=[256], class_weights=[1, 1, 1, 1]):
            # def __init__(self, data, target, dropout_prob, reg, num_hidden=[256], class_weights=pd.array([1,1,1,1])):
            self.data = data
            self.target = target
            self.dropout_prob = dropout_prob
            self.reg = reg
            self._num_hidden = num_hidden
            self._num_layers = len(num_hidden)
            self.num_classes = len(class_weights)
            self.attn_length = 0
            self.class_weights = class_weights
            self.prediction
            self.error
            self.optimize

        @lazy_property
        def make_rnn_cell(self,
                          attn_length=0,

                          base_cell=tf.nn.rnn_cell.BasicLSTMCell,

                          state_is_tuple=True):

            attn_length = self.attn_length
            input_dropout = self.dropout_prob
            output_dropout = self.dropout_prob

            cells = []
            for num_units in self._num_hidden:
                cell = base_cell(num_units, state_is_tuple=state_is_tuple)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout,
                                                     output_keep_prob=output_dropout)
                cells.append(cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

            if attn_length > 0:
                sys.path.insert(0, 'attention')
                import attention_cell_wrapper_single
                cell = attention_cell_wrapper_single.AttentionCellWrapper(
                    cell, attn_length, input_size=int(self.data.get_shape().as_list()[2]),
                    state_is_tuple=state_is_tuple)
                print
                cell
            return cell

        # predictor for slices
        @lazy_property
        def prediction(self):

            cell = self.make_rnn_cell

            # Recurrent network.
            output, final_state = tf.nn.dynamic_rnn(cell,
                                                    self.data,
                                                    dtype=tf.float32
                                                    )

            with tf.variable_scope("model") as scope:
                tf.get_variable_scope().reuse_variables()

                # final weights
                num_classes = self.num_classes
                weight, bias = self._weight_and_bias(self._num_hidden[-1], num_classes)

                # flatten + sigmoid
                if self.attn_length > 0:
                    logits = tf.matmul(final_state[0][-1][-1], weight) + bias
                else:
                    logits = tf.matmul(final_state[-1][-1], weight) + bias

                prediction = tf.nn.softmax(logits)

                return logits, prediction

        @lazy_property
        def cross_ent(self):
            predictions = self.prediction[0]
            real = tf.cast(tf.squeeze(self.target), tf.int32)

            class_weight = tf.expand_dims(tf.cast(pd.array(self.class_weights), tf.int32), axis=0)

            print("class_weights", class_weight)
            one_hot_labels = tf.cast(tf.one_hot(real, depth=self.num_classes), tf.int32)
            weight_per_label = tf.cast(tf.transpose(tf.matmul(one_hot_labels, tf.transpose(class_weight))),
                                       tf.float32)  # shape [1, batch_size]

            xent = tf.multiply(weight_per_label,
                               tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=predictions,
                                                                              name="xent_raw"))  # shape [1, batch_size]
            loss = tf.reduce_mean(xent)  # shape 1
            ce = loss
            l2 = self.reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            ce += l2
            return ce

        @lazy_property
        def optimize(self):
            learning_rate = 0.0003
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(self.cross_ent)

        @lazy_property
        def error(self):
            prediction = tf.argmax(self.prediction[1], 1)
            real = tf.cast(self.target, tf.int32)
            prediction = tf.cast(prediction, tf.int32)
            mistakes = tf.not_equal(real, prediction)
            mistakes = tf.cast(mistakes, tf.float32)
            mistakes = tf.reduce_sum(mistakes, reduction_indices=0)
            total = 128
            mistakes = tf.divide(mistakes, tf.to_float(total))
            return mistakes

        @staticmethod
        def _weight_and_bias(in_size, out_size):
            weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
            bias = tf.constant(0.1, shape=[out_size])
            return tf.Variable(weight), tf.Variable(bias)

        @lazy_property
        def summaries(self):
            tf.summary.scalar('loss', tf.reduce_mean(self.cross_ent))
            tf.summary.scalar('error', self.error)
            merged = tf.summary.merge_all()
            return merged

    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    # if attn_length > 0:
    #     # weights file initialized
    #     weight_file = 'weights.txt'
    #     with open(weight_file, 'a') as the_file:
    #         pass

    with tf.Session(config=config) as sess, tf.device('/cpu:12'):  # ROY !
        _, length, num_features = x_train.shape
        num_data_cols = num_features
        print
        "num features", num_features
        print
        "num_data cols", num_data_cols

        # with tf.Session(config=config) as sess, tf.device('/cpu:0'):
        #     _, length, num_features = x_train.shape
        #     num_data_cols = num_features
        #     print
        #     "num features", num_features
        #     print
        #     "num_data cols", num_data_cols

        # placeholders
        data = tf.placeholder(tf.float32, [None, length, num_data_cols])
        target = tf.placeholder(tf.float32, [None])
        dropout_prob = tf.placeholder(tf.float32)
        reg = tf.placeholder(tf.float32)

        # initialization
        model = VariableSequenceLabelling(data, target, dropout_prob, reg, num_hidden=NUM_HIDDEN,
                                          class_weights=class_weight)
        sess.run(tf.global_variables_initializer())
        print('Initialized Variables...')

        batch_size = BATCH_SIZE
        dp = KEEP_PROB
        rp = REGULARIZATION
        train_samples = x_train.shape[0]

        # ROY  indices =   range(train_samples
        indices = list(range(train_samples))  # ROY add List

        num_classes = NUM_CLASSES

        # for storing results
        test_data = x_test
        val_data = x_val

        val_aucs = []
        test_aucs = []
        val_aucs_macro = []
        test_aucs_macro = []

        epoch = -1

        print('Beginning Training...')

        while (epoch < 3 or max(np.diff(early_stop[-3:])) > 0):
            epoch += 1
            np.random.shuffle(indices)

            #    num_batches = train_samples // batch_size
            num_batches = train_samples // batch_size

            for batch_index in list(range(num_batches)):
                sample_indices = indices[batch_index * batch_size:batch_index * batch_size + batch_size]
                batch_data = x_train[sample_indices, :, :num_data_cols]
                batch_target = y_train[sample_indices]
                _, loss = sess.run([model.optimize, model.cross_ent],
                                   {data: batch_data, target: batch_target, dropout_prob: dp, reg: rp})


            cur_val_preds = sess.run(model.prediction, {data: x_val, target: y_val, dropout_prob: 1, reg: rp})
            val_preds = cur_val_preds[1]

            cur_test_preds = sess.run(model.prediction, {data: x_test, target: y_test, dropout_prob: 1, reg: rp})
            test_preds = cur_test_preds[1]

            val_auc_macro = roc_auc_score(y_val_classes, np.nan_to_num(val_preds), average='macro')
            test_auc_macro = roc_auc_score(y_test_classes, np.nan_to_num(test_preds), average='macro')
            val_aucs_macro.append(val_auc_macro)
            test_aucs_macro.append(test_auc_macro)

            val_auc = roc_auc_score(y_val_classes, np.nan_to_num(val_preds), average=None)
            test_auc = roc_auc_score(y_test_classes, np.nan_to_num(test_preds), average=None)
            val_aucs.append(val_auc)
            test_aucs.append(test_auc)

            if isinstance(val_aucs_macro[-1], dict):
                early_stop = [val_auc_macro for val_auc_macro in val_aucs_macro]
            else:
                early_stop = val_aucs_macro

            print
            "Val AUC = ", val_auc
            print
            "Test AUC = ", test_auc

        if isinstance(val_aucs_macro[-1], dict):
            best_epoch = np.argmax(np.array([val_auc_macro for val_auc_macro in val_aucs_macro]))
        else:
            best_epoch = np.argmax(val_aucs_macro)

        best_val_auc = val_aucs[best_epoch]
        best_test_auc = test_aucs[best_epoch]
        best_test_auc_macro = test_aucs_macro[best_epoch]

        print
        'Best Test AUC: ', best_test_auc, 'at epoch ', best_epoch
        print
        'Best Test AUC Macro: ', best_test_auc_macro, 'at epoch ', best_epoch


if __name__ == '__main__':
    pass

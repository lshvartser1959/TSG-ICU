import argparse

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from BaseFunctions import *
from simple_impute import simple_imputer


def ADAPT(X_target, Y_target, part_list=[0.001, 0.002, 0.05, 0.1], **params):
    Y_test = np.int64(Y_target)

    X_test = X_target
    y_test = Y_test

    #######
    X_test = np.nan_to_num(X_test)

    rus = RandomOverSampler(sampling_strategy='minority')
    X_test, y_test = rus.fit_sample(X_test, y_test)

    ###############################################
    # set here your prefer transformation function
    AUCs = []
    clf_just = XGBClassifier(**params)
    for part in part_list:
        X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=part, random_state=42)
        X_test1, X_test3, y_test1, y_test3 = train_test_split(X_test1, y_test1, test_size=part, random_state=42)

        clf_just.fit(X_test2, y_test2, xgb_model='model_1.model')

        y_tag = clf_just.predict(X_test3)
        y_prob_tag = clf_just.predict_proba(X_test3)[:, 0]

        acc = accuracy_score(y_tag, y_test3)
        print('source mapping function accuracy: ' + str(acc))
        s = roc_auc_score(1 - y_test3, y_prob_tag, average='macro')
        print('target auc ' + str(s))

        print('')
        AUCs.append(s)
    idx = AUCs.index(max(AUCs))
    part = part_list[idx]

    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=part, random_state=42)
    X_test1, X_test3, y_test1, y_test3 = train_test_split(X_test1, y_test1, test_size=part, random_state=42)

    clf_just.fit(X_test2, y_test2, xgb_model='model_1.model')

    y_tag = clf_just.predict(X_test1)
    y_prob_tag = clf_just.predict_proba(X_test1)[:, 0]

    acc = accuracy_score(y_tag, y_test1)
    print('source mapping function accuracy: ' + str(acc))
    s = roc_auc_score(1 - y_test1, y_prob_tag, average='macro')
    print('target auc ' + str(s))

    print('')


def run(args, GAP_TIME):
    print('Now runnimg on Gap ' + str(GAP_TIME))

    load_xys_path = args['load_xys_path']
    # GAP_TIME = args['GAP_TIME']
    tmp_xys_path = '/shared/xys/';
    # %%

    try:
        if (load_xys_path != ''):
            with open(tmp_xys_path + load_xys_path + '/x_train_concat', mode='rb') as f:
                x_train_concat = pickle.load(f)
            with open(tmp_xys_path + load_xys_path + '/x_val_concat', mode='rb') as f:
                x_val_concat = pickle.load(f)
            with open(tmp_xys_path + load_xys_path + '/x_test_concat', mode='rb') as f:
                x_test_concat = pickle.load(f)
            with open(tmp_xys_path + load_xys_path + '/y_train', mode='rb') as f:
                y_train = pickle.load(f)
            with open(tmp_xys_path + load_xys_path + '/y_val_classes', mode='rb') as f:
                y_val_classes = pickle.load(f)
            with open(tmp_xys_path + load_xys_path + '/y_test_classes', mode='rb') as f:
                y_test_classes = pickle.load(f)

            RF_LR_XGB(GAP_TIME, x_train_concat, x_val_concat, x_test_concat, y_train, y_val_classes, y_test_classes,
                      # FIX Y_VAL!!!
                      'Saved_Models/10-04-2020_2330__ourly_datah5__XGB_finalized', '', '')
        else:
            mksureDir(tmp_xys_path)
    except:
        print('no load')

    print('\nReading from file ' + DATAFILE)

    mksureDir(tmp_xys_path + runtime + '/')

    print('\nOK!\nReading vitals_labs... ')
    X = pd.read_hdf(DATAFILE, 'vitals_labs')

    X.drop('positive end-expiratory pressure set', 1 , inplace = True)
    X.drop('tidal volume set', 1, inplace=True)
    X.drop('respiratory rate set', 1, inplace=True)
    X.drop('fraction inspired oxygen set', 1, inplace=True)

    X.drop('positive end-expiratory pressure', 1, inplace=True)
    X.drop('tidal volume observed', 1, inplace=True)
    X.drop('tidal volume spontaneous', 1, inplace=True)
    X.drop('peak inspiratory pressure', 1, inplace=True)


    print('\nOK! ->>>>>' + str(X.shape[1]) + ' vitals_lab found...\n ')

    print('\nOK!\nReading interventions... ')
    Y = pd.read_hdf(DATAFILE, 'interventions')

    print('\nOK!\nReading static ')

    #static = pd.read_hdf(DATAFILE, 'patients')
    static = pd.read_csv(ARDS_HF, index_col=[0, 1, 2])

    print('\nOK!\nReading C ')

    print('\nOK!\n Preapering for stratifing .............')
    #
    first_3000 = static.index.get_level_values(0)[:3000]

    idx = pd.IndexSlice
    X_3000 = X.loc[idx[first_3000, :, :, :]]
    static_3000 = static.loc[idx[first_3000, :, :]]
    Y_3000 = Y.loc[idx[first_3000, :, :, :]]
    Y = Y_3000
    X = X_3000
    static = static_3000

    static_not_ards = static[np.logical_or(static["ARDS"] == 0, static["heart_failure"]==True)]
    static_ards = static[np.logical_and(static["ARDS"] == 1, static["heart_failure"]==False)]

    hadm_not_ards = static_not_ards.index.values
    hadm_ards = static_ards.index.values
    hadm_not_ards_first = static.index.get_level_values(0)[:]

    Y = Y[[INTERVENTION]]

    # make some temporary operation....... to see counts of vent hours per patient....
    VY = Y.reset_index()
    VY = VY.set_index('subject_id')
    VY = VY.drop('icustay_id', axis=1)
    VY = VY.drop('hadm_id', axis=1)
    yag = VY[['hours_in', 'vent']].groupby(['subject_id']).agg({'hours_in': ['count'], 'vent': ['sum']})

    # if yag > 0 so patient was ventilated
    yag = yag > 0

    # Set true or false for ventilation... in static
    static[INTERVENTION] = yag.vent.values

    # GC...
    del VY, yag

    lengths = np.array(static.reset_index().max_hours + 1).reshape(-1)

    # %%

    print('\nOK!\n ')

    print('Shape of X : ', X.shape)
    print('Shape of Y : ', Y.shape)
    print('Shape of static : ', static.shape)

    # %% md

    # Preprocessing Data

    # %% md




    # %% md

    ## Imputation and Standardization of Time Series Features

    print('\nOK!\n simple_imputer .............')
    X_clean = simple_imputer(X, hadm_not_ards_first)
    # %%

    # %%

    idx = pd.IndexSlice
    X_std = X_clean.copy()
    X_std.loc[:, idx[:, 'mean']] = X_std.loc[:, idx[:, 'mean']].apply(lambda x: minmax(x))
    X_std.loc[:, idx[:, 'time_since_measured']] = X_std.loc[:, idx[:, 'time_since_measured']].apply(
        lambda x: std_time_since_measurement(x))

    # %%

    XScols = X_std.columns.values

    X_std.columns = X_std.columns.droplevel(-1)

    # %%

    del X

    # %% md

    # %%

    # use gender, first_careunit, age and ethnicity for prediction
    static_to_keep = static[['gender', 'age', 'ethnicity', 'first_careunit']]
    static_to_keep.loc[:, 'age'] = static_to_keep['age'].apply(categorize_age)
    static_to_keep.loc[:, 'ethnicity'] = static_to_keep['ethnicity'].apply(categorize_ethnicity)
    static_to_keep = pd.get_dummies(static_to_keep, columns=['gender', 'age', 'ethnicity', 'first_careunit'])

    # %% md

    ## Create Feature Matrix

    # %%
    print('\nOK!\n X_merge .............')
    # merge time series and static data
    X_merge = pd.merge(X_std.reset_index(), static_to_keep.reset_index(), on=['subject_id', 'icustay_id', 'hadm_id'])
    X_merge = X_merge.set_index(['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])

    # time_series_col = 124  # X_merge.shape[1] - static_to_keep.shape[1] - 1 # ROY FIX - HARDCODED !! Ned to be calculated!
    time_series_col: int = int(2 + X_merge.shape[1] - static_to_keep.shape[1])
    # tsc : int = int  (2+X_merge.shape[1] - static_to_keep.shape[1])

    print('time_series_col--------------> ', time_series_col);

    del X_clean  # X_std

    subj_group0 = static_ards.index.get_level_values(0)
    subj_group1 = static_ards.index.get_level_values(1)
    subj_group2 = static_ards.index.get_level_values(2)

    idx = pd.IndexSlice
    X_ards = X_merge.loc[idx[subj_group0, subj_group2, subj_group1, :]]
    Y_ards = Y.loc[idx[subj_group0, subj_group1, subj_group2, :]]

    subj_group0 = static_not_ards.index.get_level_values(0)
    subj_group1 = static_not_ards.index.get_level_values(1)
    subj_group2 = static_not_ards.index.get_level_values(2)

    idx = pd.IndexSlice

    X_not_ards = X_merge.loc[idx[subj_group0, subj_group2, subj_group1, :]]
    Y_not_ards = Y.loc[idx[subj_group0, subj_group1, subj_group2, :]]
    x = np.array(list(X_not_ards.reset_index().groupby('subject_id').apply(create_x_matrix)))
    y = np.array(list(Y_not_ards.reset_index().groupby('subject_id').apply(create_y_matrix)))[:, :, 0]

    keys = pd.Series(X_not_ards.reset_index()['subject_id'].unique())

    # %%

    print("X tensor shape: ", x.shape)
    print("Y tensor shape: ", y.shape)
    print("lengths shape: ", lengths.shape)

    x_t = np.array(list(X_ards.reset_index().groupby('subject_id').apply(create_x_matrix)))
    y_t = np.array(list(Y_ards.reset_index().groupby('subject_id').apply(create_y_matrix)))[:, :, 0]

    # %%

    lengths_t = np.array(list(X_ards.reset_index().groupby('subject_id').apply(lambda x: x.shape[0])))

    # %%

    keys_t = pd.Series(Y_ards.reset_index()['subject_id'].unique())

    # %%

    print("X_t tensor shape: ", x_t.shape)
    print("Y_t tensor shape: ", y_t.shape)
    print("lengths shape: ", lengths_t.shape)

    # %% md


    # %% md

    # %%
    print('\nOK!\n make_3d_tensor_slices .............')
    time_series_col -= 1;

    x_train, y_train = make_3d_tensor_slices(GAP_TIME,x, y, lengths, time_series_col)
    # Remember To RELOCATE <Current supervised Y > LAST INDEX ON X on Features table and not on STATIC table !!!!  !!! Roy
    x_test, y_test = make_3d_tensor_slices(GAP_TIME,x_t, y_t, lengths_t,time_series_col)

    x_test_old = x_test
    y_test_old = y_test

    #tsc+=1;

    print('\nOK!\n label_binarize .............')

    # %%
    print('\nOK!\n label_binarize .............')
    y_train_classes = label_binarize(y_train, classes=range(NUM_CLASSES))
    y_test_classes = label_binarize(y_test, classes=range(NUM_CLASSES))
    y_test_old_classes = label_binarize(y_test_old, classes=range(NUM_CLASSES))


    # %%

    print('shape of x_train: ', x_train.shape)
    print('shape of x_test: ', x_test.shape)

    # %% md

    # Random Forest and Logistic Regression

    # %% md

    # %%

    # concatenate hourly features
    print('\nOK!\n concatenating time_series  .............')
    x_train_concat = remove_duplicate_static(x_train,time_series_col)
    x_test_concat = remove_duplicate_static(x_test,(time_series_col))
    x_test_old_concat = remove_duplicate_static(x_test_old,(time_series_col))

    # %%



    print(x_train_concat.shape)
    print(x_test_concat.shape)

    # Save TRAIN TEST SETS######################################################################################

    # PICKLES ???????
    mksureDir(tmp_xys_path + runtime+'/')

    with open(tmp_xys_path + runtime + '/x_train_concat', mode='wb') as f:
        pickle.dump(x_train_concat, f, pickle.HIGHEST_PROTOCOL)
    with open(tmp_xys_path + runtime + '/x_test_concat', mode='wb') as f:
        pickle.dump(x_test_concat, f, pickle.HIGHEST_PROTOCOL)
    with open(tmp_xys_path + runtime + '/y_train', mode='wb') as f:
        pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
    with open(tmp_xys_path + runtime + '/y_test_classes', mode='wb') as f:
        pickle.dump(y_test_classes, f, pickle.HIGHEST_PROTOCOL)

    print('\nOK!\n Train ADAPT  .............')
    np.random.seed(RANDOM)
    y_train = np.int64(y_train)
    Y_test = np.int64(y_test_classes)

    X_test = x_test_concat
    y_test = Y_test

    #######
    X_test = np.nan_to_num(X_test)

    #######
    X_train =  np.nan_to_num( x_train_concat)
    params = {'max_depth': 3, 'eta': 1,
              'learning_rate': 0.1,
              'eval_metric': 'auc', 'n_estimators': 125, 'nthread': -1,
              'n_jobs': -1, 'random_state': 34000}
    clf_just = XGBClassifier(**params)
    clf_just.fit(X_train, y_train)
    clf_just.save_model('model_1.model')

    y_tag = clf_just.predict(X_test)
    y_prob_tag = clf_just.predict_proba(X_test)[:, 0]

    acc = accuracy_score(y_tag, y_test)
    print('source mapping function accuracy: ' + str(acc))
    s = roc_auc_score(1 - y_test, y_prob_tag, average='macro')
    print('target auc ' + str(s))

    print('')

    ADAPT(x_test_concat, y_test_classes, **params)

    print('')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_xys_path', type=str, default='',
                    help='If != '' -> loads ready XYs TTV from path')

    #############
    # Parse args

    args = vars(ap.parse_args())
    print("Run with parameters :")
    for key in sorted(args.keys()):
        print(key, args[key])

    run (args, 6)

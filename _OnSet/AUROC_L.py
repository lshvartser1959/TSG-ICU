def Gntbok2(y_pred, y_true, GAP):
    # %%

    from scipy import interp
    import numpy as np
    import pandas as pd
    from itertools import cycle
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import roc_curve, auc, recall_score, confusion_matrix, precision_score, accuracy_score
    # from scipy import interp
    from sklearn.metrics import roc_auc_score
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_score = y_pred

    lw = 2
    #ONSET
    n_classes = 1
    #n_classes = y_true.shape[1]

    fpr, tpr, thr = roc_curve(y_true, y_score)
    _auc =roc_auc_score(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

    sns.set(font_scale=0.9)

    #ONSETONLY
    plt.plot(fpr, tpr, color='aqua', lw=lw,
             label='ROC curve of class OnSet \n(area = {0:0.2f})'
                   ''.format(roc_auc))

    #plt.title('Confusion matrix of', fontsize=20)


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC VENT ON_SET GAP_'+str(GAP)+'h')
    plt.legend(title='Parameter where:')
    plt.legend(bbox_to_anchor=(1, 0., 0.5, 0.5))
    # plt.legend(loc="lower right")
    plt.show()
    print('OK')

    # %%



if __name__ == '__main__':
    pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(cm, classes,
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotRoc(y, predict_test, linestyle, legend=True):
    
    labels = ['b vs l (pb/pl)', 'b vs c (pb/pc)']
    
    pb_b = predict_test[:,0] [y[:,0] == 1]
    pc_b = predict_test[:,1] [y[:,0] == 1]
    pl_b = predict_test[:,2] [y[:,0] == 1]
    
    pc_c = predict_test[:,1] [y[:,1] == 1]
    pb_c = predict_test[:,0] [y[:,1] == 1]
    
    pl_l = predict_test[:,2] [y[:,2] == 1]
    pb_l = predict_test[:,0] [y[:,2] == 1]
    
    hist_b_bvl = np.histogram( pb_b/(pb_b+pl_b), range=(0,1), bins=100000 )
    hist_l_bvl = np.histogram( pb_l/(pb_l+pl_l), range=(0,1), bins=100000 )

    hist_b_bvc = np.histogram( pb_b/(pb_b+pc_b), range=(0,1), bins=100000 )
    hist_c_bvc = np.histogram( pb_c/(pb_c+pc_c), range=(0,1), bins=100000 )

    hist_b_bvl_eff = hist_b_bvl[0][::-1].cumsum()[::-1]/hist_b_bvl[0].sum()
    hist_l_bvl_eff = hist_l_bvl[0][::-1].cumsum()[::-1]/hist_l_bvl[0].sum()
    
    hist_b_bvc_eff = hist_b_bvc[0][::-1].cumsum()[::-1]/hist_b_bvc[0].sum()
    hist_c_bvc_eff = hist_c_bvc[0][::-1].cumsum()[::-1]/hist_c_bvc[0].sum()
    
    auc_bvl = auc(hist_l_bvl_eff, hist_b_bvl_eff)
    auc_bvc = auc(hist_c_bvc_eff, hist_b_bvc_eff)
    
    plt.plot( hist_b_bvl_eff, hist_l_bvl_eff, label=f'b vs l (pb/pl), AUC = {auc_bvl:.2f}', linestyle=linestyle )
    plt.plot( hist_b_bvc_eff, hist_c_bvc_eff, label=f'b vs c (pb/pc), AUC = {auc_bvc:.2f}', linestyle=linestyle )
    
    plt.semilogy()
    plt.xlabel("b-Jet Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    if legend: plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)

def rocData(y, predict_test):
    
    labels = ['b vs l (pb/pl)', 'b vs c (pb/pc)']
    
    pb_b = predict_test[:,0] [y[:,0] == 1]
    pc_b = predict_test[:,1] [y[:,0] == 1]
    pl_b = predict_test[:,2] [y[:,0] == 1]
    
    pc_c = predict_test[:,1] [y[:,1] == 1]
    pb_c = predict_test[:,0] [y[:,1] == 1]
    
    pl_l = predict_test[:,2] [y[:,2] == 1]
    pb_l = predict_test[:,0] [y[:,2] == 1]
    
    
    hist_b_bvl = np.histogram( pb_b/(pb_b+pl_b), range=(0,1), bins=1000 )
    hist_l_bvl = np.histogram( pb_l/(pb_l+pl_l), range=(0,1), bins=1000 )

    hist_b_cvl = np.histogram( pb_b/(pb_b+pc_b), range=(0,1), bins=1000 )
    hist_c_cvl = np.histogram( pb_c/(pb_c+pc_c), range=(0,1), bins=1000 )

    hist_b_bvl_eff = hist_b_bvl.cumsum()/hist_b_bvl.sum()
    hist_l_bvl_eff = hist_l_bvl.cumsum()/hist_l_bvl.sum()
    
    hist_b_cvl_eff = hist_b_cvl.cumsum()/hist_b_cvl.sum()
    hist_c_cvl_eff = hist_c_cvl.cumsum()/hist_c_cvl.sum()

    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:,i]
        df[label + '_pred'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return effs, auc1

def makeRoc(y, predict_test, linestyle='-', legend=True):
      
    plotRoc(y, predict_test, linestyle, legend)
    
#     effs, auc1 = rocData(y, predict_test, labels)
#     plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend)
#     return predict_test

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))
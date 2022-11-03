import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def cat_max_id(catalogue,z_min,z_max):
    """ returns an int, the max number of galaxies you have in the color-cut catalogue. You have two possible catalogues
        - W1 = '/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt'
        - W4 = '/content/drive/MyDrive/GraphoVerse/W4_PHOT-SPEC_MATCH_PDR.txt'
     """
    # find number of galaxies in W1 dataset
    W = np.loadtxt(catalogue,usecols=(1,2,4,5,6,7,9,11,13,15,17))
    sel_flg = ((W[:,3]>2.)&(W[:,3]<10.))|((W[:,3]>12.)&(W[:,3]<20.))|((W[:,3]>22.)&(W[:,3]<30.))|((W[:,3]>212.)&(W[:,3]<220.))
    W = W[sel_flg]   # high quality flag
    sel_z = (W[:,2]>z_min) & (W[:,2]<z_max)
    W = W[sel_z]
    nids = len(W)
    return nids


def VIP_norm(z_min, z_max):
    """ find vipers data normalization, so that colors
        have range between [0,1]
    """

    # find global normalization over all VIPERS catalogues with good spectroscopy
    W1 = np.loadtxt('/content/drive/MyDrive/GraphoVerse/W1_PHOT-SPEC_MATCH_PDR.txt',usecols=(1,2,4,5,6,7,9,11,13,15,17))
    W4 = np.loadtxt('/content/drive/MyDrive/GraphoVerse/W4_PHOT-SPEC_MATCH_PDR.txt',usecols=(1,2,4,5,6,7,9,11,13,15,17))
    W = np.vstack((W1,W4))
    sel_flg = ((W[:,3]>2.)&(W[:,3]<10.))|((W[:,3]>12.)&(W[:,3]<20.))|((W[:,3]>22.)&(W[:,3]<30.))|((W[:,3]>212.)&(W[:,3]<220.))
    W = W[sel_flg]   # high quality flag
    sel_z = (W[:,2]>z_min) & (W[:,2]<z_max)
    W = W[sel_z]
    
    norm5  = np.ptp(W[:,5]) 
    norm6  = np.ptp(W[:,6])
    norm7  = np.ptp(W[:,7])
    norm8  = np.ptp(W[:,8])
    norm9  = np.ptp(W[:,9])
    norm10 = np.ptp(W[:,10])
    
    min5  = np.min(W[:,5]) 
    min6  = np.min(W[:,6])
    min7  = np.min(W[:,7])
    min8  = np.min(W[:,8])
    min9  = np.min(W[:,9])
    min10 = np.min(W[:,10])
    
    norms = [norm5, norm6, norm7, norm8, norm9, norm10, min5, min6, min7, min8, min9, min10]

    return norms 

def z_metrics(z_real, z_measured):
  """ Metrics of Salvato+19
      Computes the bias, the asbolute bias, the outliers fraction and the std of the measured redshift sample.
     -z_real
     -z_measured
  """

  #BIAS: mean(z_phot - z_spec)
  bias = np.mean(np.abs(z_measured - z_real))
  bias_na = np.mean(z_measured - z_real)

  #OUTLIERS FRACTION: #objects that satisfy abs(z_phot - z_spec) / (1 + z_spec) > 0.15
  N = len(z_real)
  out = np.where(np.abs(z_measured - z_real) / (1. + z_real) > 0.15 )

  #PRECISION: std((z_phot - z_spec) / (1 + z_spec)) 
  p = np.std((z_measured - z_real) / (1. + z_real))

  return [bias, bias_na, (len(out[0]) / N), p]

def best_scores(test_results, len_id, n_nbrs):
    """ test_results is the output of the evaluate helper function.
        This function yields the best scores of the neighbours of each photometric galaxies.
        It returns, in order
        nz: the redshift of the neighbour
        tz: the redshift of the target
        sc: the score prediction
        lb: the label of the pair
     """
    # all NN
    nz = test_results[:,0].reshape(len_id,n_nbrs-1) # near-z
    tz = test_results[:,1].reshape(len_id,n_nbrs-1) # true-z
    sc = test_results[:,2].reshape(len_id,n_nbrs-1) # score
    lb = test_results[:,3].reshape(len_id,n_nbrs-1) # label

    amax = np.argmax(sc,-1)
    sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
    tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
    nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
    lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

    return nz, tz, sc, lb


def find_thres(labels, scores):
    """ function to find the threshold of the classifier """
    fpr, tpr, ths = roc_curve(np.squeeze(labels).astype(np.int32), np.squeeze(scores) )
    ths = np.clip(ths,0.0,1.0)
    auc_roc = auc(fpr, tpr)
    
    # Find the optimal threshold
    mean = np.sqrt(tpr * (1 - fpr))
    #mean = tpr - fpr # youdenJ
    index = np.argmax(mean)
    thresholdOpt = round(ths[index], ndigits = 4)
    meanOpt = round(mean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)

    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, meanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    return fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt

def plot_results(Dz, data_te, th, *results, folder_img=None):
    ''' plot results and statistics '''
    nz, tz, sc, lb = results[0]
    z_c_photo = np.array([ data_te.W[id[0],4] for id in data_te.indices[data_te.ids] ])
    z_c = np.array([ data_te.W[id[0],2] for id in data_te.indices[data_te.ids] ])

    frac = len(sc[sc>th])/len(sc)

    fig, ax = plt.subplots(figsize=(6*3,6*1),ncols=3,nrows=1)
    for i in range(3):
        ax[i].tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
            grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
        ax[i].patch.set_edgecolor('black')
        ax[i].patch.set_linewidth('1')
        ax[i].set_ylim([0.4,1.2])
        ax[i].set_xlim([0.4,1.2])

    # original data
    ax[0].scatter(z_c_photo, z_c, s=0.3)
    ax[0].set_xlabel('z photo', fontsize='xx-large')
    ax[0].set_ylabel('z spec', fontsize='xx-large')
    ax[0].set_title('original', fontsize='xx-large')
    abs_bias, bias, outs, prec = z_metrics(z_c,z_c_photo)
    textstr = '\n'.join((
        r'$\sigma=%.3f$' % (prec, ),
        r'$\mathrm{out}=%.1f$ %%' % (outs*100, ),
        r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
        r'$\mathrm{b}=%.3f$' % (bias, ),
        r'$\mathrm{fr}=%.1f$ %%' % (100.0, ),)
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14,
      verticalalignment='top', bbox=props)

    # clean photo
    ax[1].scatter(z_c_photo[np.squeeze(sc)>th], z_c[np.squeeze(sc)>th], s=0.3)
    ax[1].set_title('clean', fontsize='xx-large')
    ax[1].set_xlabel('z photo', fontsize='xx-large')
    ax[1].set_ylabel('z spec', fontsize='xx-large')
    abs_bias, bias, outs, prec = z_metrics(z_c[np.squeeze(sc)>th],z_c_photo[np.squeeze(sc)>th])
    textstr = '\n'.join((
        r'$\sigma=%.3f$' % (prec, ),
        r'$\mathrm{out}=%.2f$ %%' % (outs*100, ),
        r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
        r'$\mathrm{b}=%.4f$' % (bias, ),
        r'$\mathrm{fr}=%.1f$ %%' % (frac*100, ),)
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
      verticalalignment='top', bbox=props)

    # graph estimate
    ax[2].scatter(nz[sc>th],tz[sc>th],s=0.3)
    ax[2].set_title('best z spec neighbour', fontsize='xx-large')
    ax[2].set_xlabel(r'best $z_{\mathrm{NN}}$', fontsize='xx-large')
    ax[2].set_ylabel('z spec', fontsize='xx-large')
    abs_bias, bias, outs, prec = z_metrics( tz[sc>th], nz[sc>th])
    textstr = '\n'.join((
        r'$\sigma=%.3f$' % (prec, ),
        r'$\mathrm{out}=%.2f$ %%' % (outs*100, ),
        r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
        r'$\mathrm{b}=%.4f$' % (bias, ),
        r'$\mathrm{fr}=%.1f$ %%' % (frac*100, ),)
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[2].text(0.05, 0.95, textstr, transform=ax[2].transAxes, fontsize=14,
      verticalalignment='top', bbox=props)
    if folder_img is not None:
        plt.savefig(folder_img+'/results.png',dpi=300, layout='tight')
    plt.show()

    return

def plot_roc(fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt, folder_img=None):
    plt.figure(figsize=(6*1.62,6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="AUC = %0.2f, th=%0.2f" % (auc_roc, thresholdOpt) )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fprOpt, tprOpt,  marker="o", markersize=10, markerfacecolor="green") 
    plt.hlines(1,0.,1.0,linestyles='--')
    plt.xlabel("False Positive Rate",fontsize='x-large')
    plt.ylabel("True Positive Rate",fontsize='x-large')
    plt.title("ROC")
    plt.legend(loc="lower right", fontsize='x-large')
    if folder_img is not None:
      plt.savefig(folder_img + '/roc.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_roc_dz(size_te, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot roc for list of test results, for various Dz.
        *results: unpack the list of strings directing to test results.
        size_te: number of photometric galaxies used for the test.
    '''

    fig, ax = plt.subplots(figsize=(1.62*6,6))
    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    for file_result in results:
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs, string_dz = file_result.split('_')[-2:]
        n_nbrs = int(string_nbrs.strip('nbr'))
        dz   = float(string_dz.strip('Dz'))

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs)

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt = find_thres(lb, sc)

        ax.plot(fpr, tpr, lw=2, label=" $\Delta z$=%0.2f," %dz + " AUC = %0.2f," % auc_roc + " thr=%0.2f" % thresholdOpt)
        ax.plot(fprOpt, tprOpt,  marker="o", markersize=8, markerfacecolor='darkorange',markeredgecolor='k')

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.hlines(1,0.,1.0,linestyles='--')
    ax.set_xlabel("False Positive Rate",fontsize='xx-large')
    ax.set_ylabel("True Positive Rate",fontsize='xx-large')
    ax.set_title("ROC", fontsize='xx-large')
    plt.legend(loc="lower right", fontsize='x-large')
    plt.savefig(folder+'/roc_dz.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_roc_nn(size_te, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot roc for list of test results, for various n_nbrs.
        *results: unpack the list of strings directing to test results.
        size_te: number of photometric galaxies used for the test.
    '''

    fig, ax = plt.subplots(figsize=(1.62*6,6))
    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    for file_result in results:
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs = file_result.split('_')[-2]
        n_nbrs = int(string_nbrs.strip('nbr'))

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs) 

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt = find_thres(lb, sc)

        ax.plot(fpr, tpr, lw=2, label="nbrs=%2d" %n_nbrs + " AUC = %0.2f," % auc_roc + " thr=%0.2f" % thresholdOpt)
        ax.plot(fprOpt, tprOpt,  marker="o", markersize=8, markerfacecolor='darkorange',markeredgecolor='k')

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.hlines(1,0.,1.0,linestyles='--')
    ax.set_xlabel("False Positive Rate",fontsize='xx-large')
    ax.set_ylabel("True Positive Rate",fontsize='xx-large')
    ax.set_title("ROC", fontsize='xx-large')
    plt.legend(loc="lower right", fontsize='x-large')
    plt.savefig(folder+'/roc_nn.png',dpi=300, layout='tight')
    plt.show()
    return 


def plot_results_dz(size_te, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot results of metrics with different dz
        Usage
        _args = [test_results001, test_results002, test_results004]
        dz_args = [0.01, 0.02, 0.04]
        plot_dz(dz_args,*_args)
 '''

    nplots = len(results)
    fig, ax = plt.subplots(figsize=(6*nplots,6),ncols=nplots,nrows=1)

    for j, file_result in enumerate(results):
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs, string_dz = file_result.split('_')[-2:]
        n_nbrs = int(string_nbrs.strip('nbr'))
        dz   = float(string_dz.strip('Dz'))       

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs)

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt = find_thres(lb, sc)
        th = thresholdOpt

        ax[j].scatter(nz[sc>th],tz[sc>th],s=0.3)
        ax[j].set_title(r'$\Delta z=$%.2f' %dz, fontsize='xx-large')
        ax[j].set_xlabel(r'best $z_{\mathrm{NN}}$', fontsize='xx-large')
        ax[j].set_ylabel('z spec', fontsize='xx-large')

        frac = len(sc[sc>th])/len(sc)
        abs_bias, bias, outs, prec = z_metrics( tz[sc>th], nz[sc>th])

        textstr = '\n'.join((
            r'$\sigma=%.3f$' % (prec, ),
            r'$\mathrm{out}=%.2f$ %%' % (outs*100, ),
            r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
            r'$\mathrm{b}=%.4f$' % (bias, ),
            r'$\mathrm{fr}=%.1f$ %%' % (frac*100, ),)
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[j].text(0.05, 0.95, textstr, transform=ax[j].transAxes, fontsize=14,
          verticalalignment='top', bbox=props)

        ax[j].tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
            grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
        ax[j].patch.set_edgecolor('black')
        ax[j].patch.set_linewidth('1')
        ax[j].set_ylim([0.4,1.2])
        ax[j].set_xlim([0.4,1.2])

    plt.savefig(folder+'/results_dz.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_results_nn(size_te, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot results of metrics with different nn
        Usage
        _args = [test_results001, test_results002, test_results004]
        dz_args = [0.01, 0.02, 0.04]
        plot_dz(dz_args,*_args)
 '''

    nplots = len(results)
    fig, ax = plt.subplots(figsize=(6*nplots,6),ncols=nplots,nrows=1)

    for j, file_result in enumerate(results):
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs = file_result.split('_')[-2]
        n_nbrs = int(string_nbrs.strip('nbr'))

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs)

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        fpr, tpr, auc_roc, fprOpt, tprOpt, thresholdOpt = find_thres(lb, sc)
        th = thresholdOpt

        ax[j].scatter(nz[sc>th],tz[sc>th],s=0.3)
        ax[j].set_title(r'$\mathrm{nbrs}=$%2d' %n_nbrs, fontsize='xx-large')
        ax[j].set_xlabel(r'best $z_{\mathrm{NN}}$', fontsize='xx-large')
        ax[j].set_ylabel('z spec', fontsize='xx-large')

        frac = len(sc[sc>th])/len(sc)
        abs_bias, bias, outs, prec = z_metrics( tz[sc>th], nz[sc>th])

        textstr = '\n'.join((
            r'$\sigma=%.3f$' % (prec, ),
            r'$\mathrm{out}=%.2f$ %%' % (outs*100, ),
            r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
            r'$\mathrm{b}=%.4f$' % (bias, ),
            r'$\mathrm{fr}=%.1f$ %%' % (frac*100, ),)
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[j].text(0.05, 0.95, textstr, transform=ax[j].transAxes, fontsize=14,
          verticalalignment='top', bbox=props)

        ax[j].tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
            grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
        ax[j].patch.set_edgecolor('black')
        ax[j].patch.set_linewidth('1')
        ax[j].set_ylim([0.4,1.2])
        ax[j].set_xlim([0.4,1.2])

    plt.savefig(folder+'/results_nn.png',dpi=300, layout='tight')
    plt.show()
    return

def ccplot(data_te, th, *results, folder_img=None):
    ''' '''
    nz, tz, sc, lb = results[0]
    z_c_photo = np.array([ data_te.W[id[0],4] for id in data_te.indices[data_te.ids] ])
    z_c = np.array([ data_te.W[id[0],2] for id in data_te.indices[data_te.ids] ])
    
    u_g = np.array([ data_te.W[id[0],5] - data_te.W[id[0],6] for id in data_te.indices[data_te.ids] ])
    r_i  = np.array([ data_te.W[id[0],7] - data_te.W[id[0],8] for id in data_te.indices[data_te.ids] ])
    
    out = np.where(np.abs(z_c_photo[np.squeeze(sc)>th] - z_c[np.squeeze(sc)>th]) / (1. + z_c[np.squeeze(sc)>th]) > 0.15 )
    not_out = np.where(np.abs(z_c_photo[np.squeeze(sc)>th] - z_c[np.squeeze(sc)>th]) / (1. + z_c[np.squeeze(sc)>th]) <= 0.15 )

    plt.scatter(u_g[not_out], r_i[not_out], marker='.', label="targets")
    plt.scatter(u_g[out], r_i[out], marker='.', label="outliers")
 
    plt.legend()
    plt.xlabel("$u-g$")
    plt.ylabel("$r-i$")
    plt.tight_layout()
    
    if folder_img is not None:
        plt.savefig(folder_img+'/ccplot.png', dpi=150)
    plt.show()

    return

def PDFs(size_te, labels, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot results of metrics, of different files
    '''
    from scipy.stats import skew
    from sklearn.metrics import roc_curve, auc
    nplots = len(results)
    fig, ax = plt.subplots(figsize=(6*1.62,6),ncols=1,nrows=1)

    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
       grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')
    ax.set_xlim([0.4,1.2])

    for j, file_result in enumerate(results):
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs, string_dz = file_result.split('_')[-2:]
        n_nbrs = int(string_nbrs.strip('nbr'))
        dz   = float(string_dz.strip('Dz'))       

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs)

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        th = find_thres(lb, sc)[-1]

        ax.hist(nz[sc>th],range=[0.3,1.2], bins=61,label=labels[j],histtype='step', lw=3.0)

    ax.hist(tz[sc>th],range=[0.3,1.2], bins=61,label='true z spec',ls='-',histtype='step', lw=3.0)
    ax.legend(fontsize='xx-large')
    ax.set_xlabel('redshift',fontsize='xx-large')
    ax.set_ylabel('counts',fontsize='xx-large')

    plt.savefig(folder+'/PDF_dense_v_mp.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_results_single(size_te, files, title=None, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot results of metrics, of different files
    '''

    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(6,6),ncols=1,nrows=1)

    test_results = np.load(files,allow_pickle=True)
    file_result = files[:-4]
    string_nbrs, string_dz = file_result.split('_')[-2:]
    n_nbrs = int(string_nbrs.strip('nbr'))
    dz   = float(string_dz.strip('Dz'))       

    # all NN
    nz = test_results[:,0].reshape(size_te,n_nbrs)
    tz = test_results[:,1].reshape(size_te,n_nbrs)
    sc = test_results[:,2].reshape(size_te,n_nbrs)
    lb = test_results[:,3].reshape(size_te,n_nbrs)

    # best NN
    amax = np.argmax(sc,-1)
    sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
    tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
    nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
    lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

    fpr, tpr, ths = roc_curve(np.squeeze(lb).astype(np.int32), np.squeeze(sc) )
    ths = np.clip(ths,0.0,1.0)
    roc_auc = auc(fpr, tpr)

    # find the optimal threshold
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(ths[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)
    fpropt = round(fpr[index], ndigits = 4)
    tpropt = round(tpr[index], ndigits = 4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('fpr: {}, tpr: {}'.format(fpropt, tpropt))

    th = thresholdOpt

    print('fraction of clean data=', len(sc[sc>th])/len(sc))

    ax.scatter(nz[sc>th],tz[sc>th],s=0.3)
    if title is not None:
      ax.set_title(title, fontsize='xx-large')
    ax.set_xlabel(r'best $z_{\mathrm{NN}}$', fontsize='xx-large')
    ax.set_ylabel('z spec', fontsize='xx-large')

    print('\ngraph results with optimal threshold %.2g:' %th)
    frac = len(sc[sc>th])/len(sc)
    print('fraction of clean data=', len(sc[sc>th])/len(sc))
    abs_bias, bias, outs, prec = z_metrics( tz[sc>th], nz[sc>th])

    textstr = '\n'.join((
        r'$\sigma=%.3f$' % (prec, ),
        r'$\mathrm{out}=%.2f$ %%' % (outs*100, ),
        r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
        r'$\mathrm{b}=%.4f$' % (bias, ),
        r'$\mathrm{fr}=%.1f$ %%' % (frac*100, ),)
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
      verticalalignment='top', bbox=props)

    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')
    ax.set_ylim([0.4,1.2])
    ax.set_xlim([0.4,1.2])

    plt.savefig(folder+'/input_theta.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_results_compare(size_te, *results, folder='/content/drive/MyDrive/GraphoVerse/NezNet/images'):
    ''' plot results of metrics, of different files '''

    from sklearn.metrics import roc_curve, auc
    nplots = len(results)
    fig, ax = plt.subplots(figsize=(6*nplots,6),ncols=nplots,nrows=1)

    for j, file_result in enumerate(results):
        test_results = np.load(file_result,allow_pickle=True)
        file_result = file_result[:-4]
        string_nbrs, string_dz = file_result.split('_')[-2:]
        n_nbrs = int(string_nbrs.strip('nbr'))
        dz   = float(string_dz.strip('Dz'))       

        # all NN
        nz = test_results[:,0].reshape(size_te,n_nbrs)
        tz = test_results[:,1].reshape(size_te,n_nbrs)
        sc = test_results[:,2].reshape(size_te,n_nbrs)
        lb = test_results[:,3].reshape(size_te,n_nbrs)

        # best NN
        amax = np.argmax(sc,-1)
        sc  = np.take_along_axis(sc,amax[:,None],axis=-1)
        tz  = np.take_along_axis(tz,amax[:,None],axis=-1)
        nz  = np.take_along_axis(nz,amax[:,None],axis=-1)
        lb  = np.take_along_axis(lb,amax[:,None],axis=-1)

        fpr, tpr, ths = roc_curve(np.squeeze(lb).astype(np.int32), np.squeeze(sc) )
        ths = np.clip(ths,0.0,1.0)
        roc_auc = auc(fpr, tpr)

        # find the optimal threshold
        gmean = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmean)
        thresholdOpt = round(ths[index], ndigits = 4)
        gmeanOpt = round(gmean[index], ndigits = 4)
        fpropt = round(fpr[index], ndigits = 4)
        tpropt = round(tpr[index], ndigits = 4)
        print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
        print('fpr: {}, tpr: {}'.format(fpropt, tpropt))

        th = thresholdOpt

        print('fraction of clean data=', len(sc[sc>th])/len(sc))

        ax[j].scatter(nz[sc>th],tz[sc>th],s=0.3)
        #ax[j].set_title(r'$\Delta z=$%.2f' %dz, fontsize='xx-large')
        ax[j].set_xlabel(r'best $z_{\mathrm{NN}}$', fontsize='xx-large')
        ax[j].set_ylabel('z spec', fontsize='xx-large')

        print('\ngraph results with optimal threshold %.2g:' %th)
        frac = len(sc[sc>th])/len(sc)
        print('fraction of clean data=', len(sc[sc>th])/len(sc))
        abs_bias, bias, outs, prec = z_metrics( tz[sc>th], nz[sc>th])

        textstr = '\n'.join((
            r'$\mathrm{fr}=%.3f$' % (frac, ),
            r'$\mathrm{out}=%.4f$' % (outs, ),
            r'$|\mathrm{b}|=%.3f$' % (abs_bias, ),
            r'$\mathrm{b}=%.4f$' % (bias, ),
            r'$\mathrm{prec}=%.3f$' % (prec, ),)
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[j].text(0.05, 0.95, textstr, transform=ax[j].transAxes, fontsize=14,
          verticalalignment='top', bbox=props)

        ax[j].tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
            grid_alpha=0.5,labelsize='x-large', which ='both',top=True,right=True)
        ax[j].patch.set_edgecolor('black')
        ax[j].patch.set_linewidth('1')
        ax[j].set_ylim([0.4,1.2])
        ax[j].set_xlim([0.4,1.2])

    #plt.savefig(folder+'/results_dz.eps',dpi=150, layout='tight', format='eps')
    plt.show()
    return


def plot_Dz_gradients(*list_weights, folder ='/content/drive/MyDrive/GraphoVerse/NezNet/'):
    ''' compare nearest neighbours gradients for the same model, with different Dz '''

    labels = ['$z_{\mathrm{spec}}$', 'u', 'g', 'r', 'i', 'z', 'Ks']

    n = len(list_weights)
    width = 1/n/1.5       # width of the bars
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=(6*1.62,6))
    ax.set_title('gradients of best neighbours',fontsize='xx-large')
    ax.set_xticks(np.arange(len(labels))+width)
    ax.set_xticklabels(labels)
    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='xx-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    for i, weights in enumerate(list_weights):
      if not os.path.exists(folder + 'data/grad_' + weights + '.npy'):
        print(folder + 'data/grad_' + weights + '.npy')
        raise ValueError('gradients for your model do not exist')

      string_Dz = weights.split('_')[-1]
      string_Dz = string_Dz.strip('Dz')
      Dz = float(string_Dz)
      grads = np.load(folder + 'data/grad_' + weights + '.npy', allow_pickle=True)

      grads_nn = grads[1]/grads[1][0] # divide by the gradients of zspec
      ax.bar(x + i*width, grads_nn, width, label='$\Delta z=%.2f$' %Dz )

    ax.legend(fontsize='xx-large')
    fig.tight_layout()
    plt.savefig(folder+'images/grad_Dz.png',dpi=300, layout='tight')
    plt.show()
    return

def plot_pair_gradients(weights, folder ='/content/drive/MyDrive/GraphoVerse/NezNet/'):

    if not os.path.exists(folder + 'data/grad_' + weights + '.npy'):
      print(folder + 'data/grad_' + weights + '.npy')
      raise ValueError('gradients for your model do not exist')

    grads = np.load(folder + 'data/grad_' + weights + '.npy', allow_pickle=True)

    if weights.startswith('bs') or weights.startswith('neznet') or weights.startswith('dense'):
      labels = ['z_spec', 'u', 'g', 'r', 'i', 'z', 'Ks']

    if weights.startswith('hav'):
      labels = ['z_spec', 'u', 'g', 'r', 'i', 'z', 'Ks', '$\Delta \Theta$']

    elif weights.startswith('radec'):
      labels = ['z_spec', 'u', 'g', 'r', 'i', 'z', 'Ks', 'RA', 'DEC']

    else:
      pass

    print(grads.shape)
    pos_g_c  = grads[0]
    pos_g_nn = grads[1]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(5*1.62,5))

    rects1 = ax.bar(x - width/2, pos_g_c, width, label='Central')
    rects2 = ax.bar(x + width/2, pos_g_nn, width, label='Nearest Neighbour')

    ax.set_title('gradients of best scores',fontsize='xx-large')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(fontsize='xx-large')
    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='xx-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')
    fig.tight_layout()
    plt.savefig(folder+'images/grad_pair_'+weights+'.png',dpi=300, layout='tight')
    plt.show()
    return


def plot_neigh_gradients(weights, folder ='/content/drive/MyDrive/GraphoVerse/NezNet/'):

    if not os.path.exists(folder + 'data/grad_' + weights + '.npy'):
      print(folder + 'data/grad_' + weights + '.npy')
      raise ValueError('gradients for your model do not exist')

    grads = np.load(folder + 'data/grad_' + weights + '.npy', allow_pickle=True)

    if weights.startswith('bs') or weights.startswith('neznet') or weights.startswith('dense'):
      labels = ['$z_{\mathrm{spec}}$', 'u', 'g', 'r', 'i', 'z', 'Ks']

    if weights.startswith('hav'):
      labels = ['$z_{\mathrm{spec}}$', 'u', 'g', 'r', 'i', 'z', 'Ks', r'$\Delta \Theta$']

    elif weights.startswith('radec'):
      labels = ['$z_{\mathrm{spec}}$', 'u', 'g', 'r', 'i', 'z', 'Ks', 'RA', 'DEC']

    else:
      pass

    _  = grads[0]
    pos_g_nn = grads[1]

    x = np.arange(len(labels))# the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(5*1.62,5))

    rects2 = ax.bar(x , pos_g_nn, width, label='Nearest Neighbour')

    ax.set_title('gradients of best scores',fontsize='xx-large')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    #ax.legend(fontsize='xx-large')
    ax.tick_params( direction='in', length=6, width=2, colors='k',grid_color='k',
        grid_alpha=0.5,labelsize='xx-large', which ='both',top=True,right=True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')
    fig.tight_layout()
    plt.savefig(folder+'images/grad_neigh_'+weights+'.png',dpi=300, layout='tight')
    plt.show()
    return


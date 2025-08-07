import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import binned_statistic


def convertXY2PtPhi(arrayXY):
    nevents = arrayXY.shape[0]
    arrayPtPhi = np.zeros((nevents, 2))
    arrayPtPhi[:, 0] = np.sqrt((arrayXY[:, 0] ** 2 + arrayXY[:, 1] ** 2))
    arrayPtPhi[:, 1] = np.arctan2(arrayXY[:, 1], arrayXY[:, 0])
    return arrayPtPhi


def generate_all_plots(Yr_test, predict_test, PUPPI_pt, path_out):


    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=path_out)
    
    # Convert XY to PtPhi format for additional plots
    Yr_test_ptphi = convertXY2PtPhi(Yr_test)
    predict_test_ptphi = convertXY2PtPhi(predict_test)
    PUPPI_pt_ptphi = convertXY2PtPhi(PUPPI_pt)
    
    # Generate error analysis plots
    make_error_plots(predict_test_ptphi, PUPPI_pt_ptphi, Yr_test_ptphi, path_out)
    
    # Save prediction arrays for further analysis
    extract_result(predict_test, Yr_test, path_out, "TTbar", "ML")
    extract_result(PUPPI_pt, Yr_test, path_out, "TTbar", "PU")
    
    print(f"All plots generated and saved to {path_out}")


def MakePlots(trueXY, mlXY, puppiXY, path_out):

    plt.style.use(hep.style.CMS)

    true_ptPhi = convertXY2PtPhi(trueXY)
    ml_ptPhi = convertXY2PtPhi(mlXY)
    puppi_ptPhi = convertXY2PtPhi(puppiXY)

    # Generate 1D histograms
    Make1DHists(
        trueXY[:, 0], mlXY[:, 0], puppiXY[:, 0],
        -400, 400, 40, False, "MET X [GeV]", "A.U.",
        f"{path_out}MET_x.png",
    )
    Make1DHists(
        trueXY[:, 1], mlXY[:, 1], puppiXY[:, 1],
        -400, 400, 40, False, "MET Y [GeV]", "A.U.",
        f"{path_out}MET_y.png",
    )
    Make1DHists(
        true_ptPhi[:, 0], ml_ptPhi[:, 0], puppi_ptPhi[:, 0],
        0, 400, 40, False, "MET Pt [GeV]", "A.U.",
        f"{path_out}MET_pt.png",
    )

    # Calculate binned statistics for response analysis
    nbins = 20
    binnings = np.linspace(0, 400, num=nbins + 1)
    phiBinnings = np.linspace(-3.15, 3.15, num=nbins + 1)
    
    truth_means, bin_edges, binnumber = binned_statistic(
        true_ptPhi[:, 0], true_ptPhi[:, 0], statistic="mean",
        bins=binnings, range=(0, 400),
    )
    ml_means, _, _ = binned_statistic(
        true_ptPhi[:, 0], ml_ptPhi[:, 0], statistic="mean",
        bins=binnings, range=(0, 400),
    )
    puppi_means, _, _ = binned_statistic(
        true_ptPhi[:, 0], puppi_ptPhi[:, 0], statistic="mean",
        bins=binnings, range=(0, 400),
    )

    # Generate response plot
    plt.figure()
    plt.hlines(
        truth_means / truth_means, bin_edges[:-1], bin_edges[1:],
        colors="k", lw=5, label="Truth", linestyles="solid",
    )
    plt.hlines(
        ml_means / truth_means, bin_edges[:-1], bin_edges[1:],
        colors="r", lw=5, label="ML", linestyles="solid",
    )
    plt.hlines(
        puppi_means / truth_means, bin_edges[:-1], bin_edges[1:],
        colors="g", lw=5, label="PUPPI", linestyles="solid",
    )
    plt.xlim(0, 400.0)
    plt.ylim(0, 1.1)
    plt.xlabel("Truth MET [GeV]")
    plt.legend(loc="lower right")
    plt.ylabel("<MET Estimation>/<MET Truth>")
    plt.savefig(f"{path_out}MET_response.png")
    plt.close()

    # Calculate resolution metrics
    def resolqt(y):
        return (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0

    # Response correction factors
    responseCorrection_ml = np.take(
        truth_means / ml_means, np.digitize(true_ptPhi[:, 0], binnings) - 1, mode="clip"
    )
    responseCorrection_puppi = np.take(
        truth_means / puppi_means,
        np.digitize(true_ptPhi[:, 0], binnings) - 1, mode="clip",
    )

    # Phi difference calculations
    Phi_diff_ml = true_ptPhi[:, 1] - ml_ptPhi[:, 1]
    Phi_diff_ml = np.where(Phi_diff_ml < -math.pi, Phi_diff_ml + 2 * math.pi, Phi_diff_ml)
    Phi_diff_ml = np.where(Phi_diff_ml > math.pi, Phi_diff_ml - 2 * math.pi, Phi_diff_ml)

    Phi_diff_puppi = true_ptPhi[:, 1] - puppi_ptPhi[:, 1]
    Phi_diff_puppi = np.where(Phi_diff_puppi < -math.pi, Phi_diff_puppi + 2 * math.pi, Phi_diff_puppi)
    Phi_diff_puppi = np.where(Phi_diff_puppi > math.pi, Phi_diff_puppi - 2 * math.pi, Phi_diff_puppi)

    # Calculate resolution metrics in bins
    bin_resolX_ml, bin_edges, binnumber = binned_statistic(
        true_ptPhi[:, 0], trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolY_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0], trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolPt_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0], true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolPhi_ml, bin_edgesPhi, binnumberPhi = binned_statistic(
        true_ptPhi[:, 1], Phi_diff_ml, statistic=resolqt,
        bins=phiBinnings, range=(-3.15, 3.15),
    )

    bin_resolX_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0], trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolY_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0], trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolPt_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0], true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi,
        statistic=resolqt, bins=binnings, range=(0, 400),
    )
    bin_resolPhi_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 1], Phi_diff_puppi, statistic=resolqt,
        bins=phiBinnings, range=(-3.15, 3.15),
    )

    # Calculate resolution magnitudes and differences
    bin_resolXYmagnitude_ml = (bin_resolX_ml**2 + bin_resolY_ml**2) ** 0.5
    bin_resolXYmagnitude_puppi = (bin_resolX_puppi**2 + bin_resolY_puppi**2) ** 0.5
    bin_resolXYmagnitude_difference = bin_resolXYmagnitude_puppi - bin_resolXYmagnitude_ml
    bin_resolPt_difference = bin_resolPt_puppi - bin_resolPt_ml

    # Calculate overall resolution metrics
    average_xRes_ml = resolqt(trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml)
    average_yRes_ml = resolqt(trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml)
    average_ptRes_ml = resolqt(true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml)

    average_xRes_puppi = resolqt(trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi)
    average_yRes_puppi = resolqt(trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi)
    average_ptRes_puppi = resolqt(true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi)

    averageXYmag_Res_puppi = (average_xRes_puppi**2 + average_yRes_puppi**2) ** (0.5)
    averageXYmag_Res_ml = (average_xRes_ml**2 + average_yRes_ml**2) ** (0.5)
    averageXYmag_Res_difference = averageXYmag_Res_puppi - averageXYmag_Res_ml
    averagePt_Res_difference = average_ptRes_puppi - average_ptRes_ml

    # Calculate error bars
    nEvents_inBin, _ = np.histogram(binnumber, bins=nbins, range=(1, nbins))
    rootN = np.sqrt(nEvents_inBin)
    nEvents_inBin_phi, _ = np.histogram(binnumberPhi, bins=nbins, range=(1, nbins))
    rootN_phi = np.sqrt(nEvents_inBin_phi)

    # Setup for error bar positioning
    binWidth = binnings[1]
    binCenter = binWidth / 2
    leftOfBinCenter = 0.4 * binWidth
    rightOfBinCenter = 0.6 * binWidth

    # Generate XY resolution plots
    fig1 = plt.figure(figsize=(15, 12), tight_layout=True)
    plt.subplots_adjust(wspace=0.2, hspace=0)

    # X resolution plot
    ax11 = fig1.add_subplot(2, 2, 1)
    ax11.hlines(bin_resolX_ml, bin_edges[:-1], bin_edges[1:],
                colors="r", lw=3, label="ML", linestyles="solid")
    ax11.hlines(bin_resolX_puppi, bin_edges[:-1], bin_edges[1:],
                colors="g", lw=3, label="PUPPI", linestyles="solid")
    ax11.errorbar(bin_edges[:-1] + rightOfBinCenter, bin_resolX_ml,
                  yerr=bin_resolX_ml / rootN, fmt="none", color="r")
    ax11.errorbar(bin_edges[:-1] + leftOfBinCenter, bin_resolX_puppi,
                  yerr=bin_resolX_puppi / rootN, fmt="none", color="g")
    ax11.grid()
    ax11.set_ylabel(r"$\sigma(MET)$ [GeV]", fontsize=19)
    ax11.set_title("MET-x Resolution", fontsize=22)

    # Y resolution plot
    ax12 = fig1.add_subplot(2, 2, 2, sharey=ax11)
    ax12.hlines(bin_resolY_ml, bin_edges[:-1], bin_edges[1:],
                colors="r", lw=3, label="$\\sigma_{ML}$", linestyles="solid")
    ax12.hlines(bin_resolY_puppi, bin_edges[:-1], bin_edges[1:],
                colors="g", lw=3, label="$\\sigma_{PUPPI}$", linestyles="solid")
    ax12.errorbar(bin_edges[:-1] + rightOfBinCenter, bin_resolY_ml,
                  yerr=bin_resolY_ml / rootN, fmt="none", color="r")
    ax12.errorbar(bin_edges[:-1] + leftOfBinCenter, bin_resolY_puppi,
                  yerr=bin_resolY_puppi / rootN, fmt="none", color="g")
    ax12.legend(loc="upper center", prop={"size": 19})
    ax12.grid()
    ax12.set_title("MET-y Resolution", fontsize=22)

    # Absolute differences plot
    ax13 = fig1.add_subplot(2, 2, 3, sharex=ax11)
    ax13.hlines(bin_resolXYmagnitude_difference, bin_edges[:-1], bin_edges[1:],
                lw=5, linestyles="solid")
    ax13.axhline(y=0, color="black", linestyle="-")
    ax13.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax13.set_ylabel(r"$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]", fontsize=19)
    ax13.grid()
    ax13.set_title("Absolute Resolution Differences (PUPPI - ML)", fontsize=22)

    # Relative differences plot
    ax14 = fig1.add_subplot(2, 2, 4, sharex=ax12)
    ax14.hlines(bin_resolXYmagnitude_difference / truth_means, bin_edges[:-1], bin_edges[1:],
                lw=5, linestyles="solid")
    ax14.axhline(y=0, color="black", linestyle="-")
    ax14.set_ylim(
        min(bin_resolXYmagnitude_difference / truth_means) - 0.1,
        max(bin_resolXYmagnitude_difference / truth_means) + 0.1,
    )
    ax14.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax14.set_ylabel(r"$(\sigma_{PUPPI} - \sigma_{ML})$ / $\mu_{bin}$", fontsize=19)
    ax14.grid()
    ax14.set_title("Relative Resolution Differences", fontsize=22)

    trainingName = path_out.split("/")[-2]
    fig1.text(0, 1.06, f"training: {trainingName}", fontsize=19)
    fig1.text(
        0, 1.03,
        r"$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{xyDIF}$;  $Mean(\sigma_{xyDIF})=$"
        + f"{round(averageXYmag_Res_difference,3)}", fontsize=19,
    )

    # Generate Pt resolution plots
    fig2 = plt.figure(figsize=(15, 12), tight_layout=True)

    # Pt resolution plot
    ax21 = fig2.add_subplot(2, 2, 1)
    ax21.hlines(bin_resolPt_ml, bin_edges[:-1], bin_edges[1:],
                colors="r", lw=3, label="ML", linestyles="solid")
    ax21.hlines(bin_resolPt_puppi, bin_edges[:-1], bin_edges[1:],
                colors="g", lw=3, label="PUPPI", linestyles="solid")
    ax21.errorbar(bin_edges[:-1] + rightOfBinCenter, bin_resolPt_ml,
                  yerr=bin_resolPt_ml / rootN, fmt="none", color="r")
    ax21.errorbar(bin_edges[:-1] + leftOfBinCenter, bin_resolPt_puppi,
                  yerr=bin_resolPt_puppi / rootN, fmt="none", color="g")
    ax21.set_xlabel("truth met [gev]", fontsize=19)
    ax21.set_ylabel(r"$\sigma(MET)$ [GeV]", fontsize=20)
    ax21.grid()
    ax21.set_title("MET-pt Resolution", fontsize=22)

    # Phi resolution plot
    ax22 = fig2.add_subplot(2, 2, 2)
    ax22.hlines(bin_resolPhi_ml, bin_edgesPhi[:-1], bin_edgesPhi[1:],
                colors="r", lw=3, label="$\\sigma_{ML}$", linestyles="solid")
    ax22.hlines(bin_resolPhi_puppi, bin_edgesPhi[:-1], bin_edgesPhi[1:],
                colors="g", lw=3, label="$\\sigma_{PUPPI}$", linestyles="solid")
    ax22.errorbar(bin_edgesPhi[:-1] + 0.13, bin_resolPhi_ml,
                  yerr=bin_resolPhi_ml / rootN_phi, fmt="none", color="r")
    ax22.errorbar(bin_edgesPhi[:-1] + 0.17, bin_resolPhi_puppi,
                  yerr=bin_resolPhi_puppi / rootN_phi, fmt="none", color="g")
    ax22.set_ylabel("radian", fontsize=20)
    ax22.set_ylim(0.0, 1.0)
    ax22.grid()
    ax22.set_xlabel(r"$\phi$ angle", fontsize=19)
    ax22.legend(loc="upper center", prop={"size": 19})
    ax22.set_title(r"MET-$\Phi$ Resolution", fontsize=20)

    # Pt absolute differences plot
    ax23 = fig2.add_subplot(2, 2, 3, sharex=ax11)
    ax23.hlines(bin_resolPt_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles="solid")
    ax23.axhline(y=0, color="black", linestyle="-")
    ax23.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax23.set_ylabel(r"$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]", fontsize=20)
    ax23.grid()
    ax23.set_title("Absolute Resolution Differences", fontsize=22)

    # Pt relative differences plot
    ax24 = fig2.add_subplot(2, 2, 4, sharex=ax12)
    ax24.hlines(bin_resolPt_difference / truth_means, bin_edges[:-1], bin_edges[1:],
                lw=5, linestyles="solid")
    ax24.axhline(y=0, color="black", linestyle="-")
    ax24.set_ylim(
        min(bin_resolPt_difference / truth_means) - 0.1,
        max(bin_resolPt_difference / truth_means) + 0.1,
    )
    ax24.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax24.set_ylabel(r"$(\sigma_{PUPPI} - \sigma_{ML})$ / $\mu_{bin}$", fontsize=19)
    ax24.grid()
    ax24.set_title("Relative Resolution Differences", fontsize=22)

    fig2.text(0, 1.06, f"training: {trainingName}", fontsize=19)
    fig2.text(
        0, 1.03,
        r"$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{DIF}$;  $Mean(\sigma_{DIF})=$"
        + f"{round(averagePt_Res_difference,3)}", fontsize=19,
    )

    # Save plots
    fig1.savefig(f"{path_out}XY_resolution_plots.png", bbox_inches="tight")
    fig2.savefig(f"{path_out}pt_resolution_plots.png", bbox_inches="tight")


def Make1DHists(
    truth, ML, PUPPI, xmin=0, xmax=400, nbins=100, density=False,
    xname="pt [GeV]", yname="A.U.", outputname="1ddistribution.png",
):

    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 8))
    plt.hist(truth, bins=nbins, range=(xmin, xmax), density=density,
             histtype="step", facecolor="k", label="Truth")
    plt.hist(ML, bins=nbins, range=(xmin, xmax), density=density,
             histtype="step", facecolor="r", label="ML")
    plt.hist(PUPPI, bins=nbins, range=(xmin, xmax), density=density,
             histtype="step", facecolor="g", label="PUPPI")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


def make_error_plots(predict_test_ptphi, PUPPI_pt_ptphi, Yr_test_ptphi, path_out):

    MET_rel_error_opaque(
        predict_test_ptphi[:, 0], PUPPI_pt_ptphi[:, 0], Yr_test_ptphi[:, 0],
        name=f"{path_out}rel_error_opaque.png",
    )
    

    MET_binned_predict_mean_opaque(
        predict_test_ptphi[:, 0], PUPPI_pt_ptphi[:, 0], Yr_test_ptphi[:, 0],
        20, 0, 500, 0, ".", name=f"{path_out}PrVSGen.png",
    )
    

    Phi_abs_error_opaque(
        PUPPI_pt_ptphi[:, 1], predict_test_ptphi[:, 1], Yr_test_ptphi[:, 1],
        name=f"{path_out}Phi_abs_err.png"
    )
    
 
    Pt_abs_error_opaque(
        PUPPI_pt_ptphi[:, 0], predict_test_ptphi[:, 0], Yr_test_ptphi[:, 0],
        name=f"{path_out}Pt_abs_error.png",
    )


def MakeEdgeHist(edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"):

    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 8))
    plt.hist(edge_feat, bins=nbins, density=density, histtype="step",
             facecolor="k", label="Truth")
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


def extract_result(feat_array, targ_array, path, name, mode):

    np.save(f'{path}{name}_feature_array_{mode}MET', feat_array)
    np.save(f'{path}{name}_target_array_{mode}MET', targ_array)


def MET_rel_error_opaque(predict_met, predict_met2, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met
    mask = (rel_err[:] > 3)
    rel_err = rel_err[~mask]

    rel_err2 = (predict_met2 - gen_met)/gen_met
    mask2 = (rel_err2[:] > 3)
    rel_err2 = rel_err2[~mask2]

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3., 3., 51), label='ML', alpha=0.5, color='red')
    plt.hist(rel_err2, bins=np.linspace(-3., 3., 51), label='PUPPI', alpha=0.5, color='green')
    plt.xlabel("relative error (predict - true)/true", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.title('Relative Pt error', size=18, fontweight='bold', loc='right')
    plt.legend()
    plt.savefig(name)
    plt.close()


def MET_binned_predict_mean_opaque(predict_met, predict_met2, gen_met, binning, mini, maxi, genMET_cut, corr_check, name='predict_mean.pdf'):
    bin_ = int((maxi - mini)/binning)
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)

    for j in range(bin_):
        mask = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET[j] = np.mean(gen_met[mask])
        y_predict[j] = np.mean(predict_met[mask])
        X_error[j] = np.std(gen_met[mask])
        y_error[j] = np.std(predict_met[mask])

    X_genMET2 = np.zeros(bin_)
    X_error2 = np.zeros(bin_)
    y_predict2 = np.zeros(bin_)
    y_error2 = np.zeros(bin_)

    for j in range(bin_):
        mask2 = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET2[j] = np.mean(gen_met[mask2])
        y_predict2[j] = np.mean(predict_met2[mask2])
        X_error2[j] = np.std(gen_met[mask2])
        y_error2[j] = np.std(predict_met2[mask2])

    plt.errorbar(X_genMET2, y_predict2, xerr=X_error2, yerr=y_error2,
                 label='PUPPI MET', color='green', uplims=y_error2, lolims=y_error2)
    plt.errorbar(X_genMET, y_predict, xerr=X_error, yerr=y_error,
                 label='Predicted MET', color='red', uplims=y_error, lolims=y_error)

    # x = y plot
    X = np.arange(mini, maxi, binning)
    plt.plot(X, X, 'r-')
    
    plt.xlim(mini, maxi)
    plt.ylim(mini, maxi)
    plt.xlabel('Gen MET mean [GeV]', fontsize=16)
    plt.ylabel('predicted MET mean [GeV]', fontsize=16)
    plt.legend()
    plt.savefig(name)
    plt.close()


def Pt_abs_error_opaque(puppi_met, ml_met, gen_met, name='Met_res.pdf'):
    puppi_err = (puppi_met - gen_met)
    ml_err = (ml_met - gen_met)
    
    plt.figure()
    plt.hist(puppi_err, bins=np.linspace(-250, 250, 51), alpha=0.5, label='puppi')
    plt.hist(ml_err, bins=np.linspace(-250, 250, 51), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Pt error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.close()


def Phi_abs_error_opaque(puppi_phi, ml_phi, gen_phi, name='Met_res.pdf'):
    puppi_err = (puppi_phi - gen_phi)
    ml_err = (ml_phi - gen_phi)
    
    plt.figure()
    plt.hist(puppi_err, bins=np.linspace(-3.5, 3.5, 51), alpha=0.5, label='puppi')
    plt.hist(ml_err, bins=np.linspace(-3.5, 3.5, 51), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Phi error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.close()
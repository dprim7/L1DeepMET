import math
import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy.stats import binned_statistic
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

plt.style.use(hep.style.CMS)


def setup_cms_style():
    """Apply CMS style to plots."""
    plt.style.use(hep.style.CMS)


def convertXY2PtPhi(arrayXY):
    """Convert from array with [:,0] as X and [:,1] as Y to [:,0] as pt and [:,1] as phi."""
    nevents = arrayXY.shape[0]
    arrayPtPhi = np.zeros((nevents, 2))
    arrayPtPhi[:, 0] = np.sqrt((arrayXY[:, 0] ** 2 + arrayXY[:, 1] ** 2))
    arrayPtPhi[:, 1] = np.arctan2(arrayXY[:, 1], arrayXY[:, 0])
    return arrayPtPhi


def resolqt(y):
    """Calculate width of a distribution at 1 standard deviation."""
    return (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0


def phidiff(phi1, phi2):
    phi_r = phi1 - phi2
    if phi_r > math.pi:
        phi_r = phi_r - 2 * math.pi
    if phi_r < -math.pi:
        phi_r = phi_r + 2 * math.pi
    return phi_r 


def MakePlots(trueXY, mlXY, puppiXY, path_out):
    """
    Make the 1d distribution, response, resolution, and response-corrected resolution plots.
    Input has [:,0] as X and [:,1] as Y.
    
    Args:
        trueXY: True MET values in XY coordinates
        mlXY: ML predicted MET values in XY coordinates  
        puppiXY: PUPPI MET values in XY coordinates
        path_out: Output directory path
    """
    true_ptPhi = convertXY2PtPhi(trueXY)
    ml_ptPhi = convertXY2PtPhi(mlXY)
    puppi_ptPhi = convertXY2PtPhi(puppiXY)
    # [:,0] is pt; [:,1] is phi

    Make1DHists(
        trueXY[:, 0],
        mlXY[:, 0],
        puppiXY[:, 0],
        -400,
        400,
        40,
        False,
        "MET X [GeV]",
        "A.U.",
        f"{path_out}MET_x.png",
    )
    Make1DHists(
        trueXY[:, 1],
        mlXY[:, 1],
        puppiXY[:, 1],
        -400,
        400,
        40,
        False,
        "MET Y [GeV]",
        "A.U.",
        f"{path_out}MET_y.png",
    )
    Make1DHists(
        true_ptPhi[:, 0],
        ml_ptPhi[:, 0],
        puppi_ptPhi[:, 0],
        0,
        400,
        40,
        False,
        "MET Pt [GeV]",
        "A.U.",
        f"{path_out}MET_pt.png",
    )

    # do statistics
    nbins = 20
    binnings = np.linspace(0, 400, num=nbins + 1)  # create 20 bins for pt from 0 to 400 GeV
    phiBinnings = np.linspace(-3.15, 3.15, num=nbins + 1)
    truth_means, bin_edges, binnumber = binned_statistic(
        true_ptPhi[:, 0],
        true_ptPhi[:, 0],
        statistic="mean",
        bins=binnings,
        range=(0, 400),
    )
    ml_means, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        ml_ptPhi[:, 0],
        statistic="mean",
        bins=binnings,
        range=(0, 400),
    )
    puppi_means, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        puppi_ptPhi[:, 0],
        statistic="mean",
        bins=binnings,
        range=(0, 400),
    )

    # plot response
    plt.figure()
    plt.hlines(
        truth_means / truth_means,
        bin_edges[:-1],
        bin_edges[1:],
        colors="k",
        lw=5,
        label="Truth",
        linestyles="solid",
    )
    plt.hlines(
        ml_means / truth_means,
        bin_edges[:-1],
        bin_edges[1:],
        colors="r",
        lw=5,
        label="ML",
        linestyles="solid",
    )
    plt.hlines(
        puppi_means / truth_means,
        bin_edges[:-1],
        bin_edges[1:],
        colors="g",
        lw=5,
        label="PUPPI",
        linestyles="solid",
    )
    plt.xlim(0, 400.0)
    plt.ylim(0, 1.1)
    plt.xlabel("Truth MET [GeV]")
    plt.legend(loc="lower right")
    plt.ylabel("<MET Estimation>/<MET Truth>")
    plt.savefig(f"{path_out}MET_response.png")
    plt.close()

    # response correction factors
    responseCorrection_ml = np.take(
        truth_means / ml_means, np.digitize(true_ptPhi[:, 0], binnings) - 1, mode="clip"
    )
    responseCorrection_puppi = np.take(
        truth_means / puppi_means,
        np.digitize(true_ptPhi[:, 0], binnings) - 1,
        mode="clip",
    )

    # Phi calculation
    Phi_diff_ml = true_ptPhi[:, 1] - ml_ptPhi[:, 1]
    Phi_diff_ml = np.where(
        Phi_diff_ml < -math.pi, Phi_diff_ml + 2 * math.pi, Phi_diff_ml
    )
    Phi_diff_ml = np.where(
        Phi_diff_ml > math.pi, Phi_diff_ml - 2 * math.pi, Phi_diff_ml
    )

    Phi_diff_puppi = true_ptPhi[:, 1] - puppi_ptPhi[:, 1]
    Phi_diff_puppi = np.where(
        Phi_diff_puppi < -math.pi, Phi_diff_puppi + 2 * math.pi, Phi_diff_puppi
    )
    Phi_diff_puppi = np.where(
        Phi_diff_puppi > math.pi, Phi_diff_puppi - 2 * math.pi, Phi_diff_puppi
    )

    # compute resolutions inside all 20 bins
    bin_resolX_ml, bin_edges, binnumber = binned_statistic(
        true_ptPhi[:, 0],
        trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolY_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolPt_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolPhi_ml, bin_edgesPhi, binnumberPhi = binned_statistic(
        true_ptPhi[:, 1],
        Phi_diff_ml,
        statistic=resolqt,
        bins=phiBinnings,
        range=(-3.15, 3.15),
    )

    bin_resolX_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolY_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolPt_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi,
        statistic=resolqt,
        bins=binnings,
        range=(0, 400),
    )
    bin_resolPhi_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 1],
        Phi_diff_puppi,
        statistic=resolqt,
        bins=phiBinnings,
        range=(-3.15, 3.15),
    )

    # calculate the resolution magnitude inside all 20 bins
    bin_resolXYmagnitude_ml = (bin_resolX_ml**2 + bin_resolY_ml**2) ** 0.5
    bin_resolXYmagnitude_puppi = (bin_resolX_puppi**2 + bin_resolY_puppi**2) ** 0.5
    bin_resolXYmagnitude_difference = (
        bin_resolXYmagnitude_puppi - bin_resolXYmagnitude_ml
    )

    # transverse MET resolution difference
    bin_resolPt_difference = bin_resolPt_puppi - bin_resolPt_ml

    # compute the resolution over the entire dataset (1 bin)
    average_xRes_ml = resolqt(trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml)
    average_yRes_ml = resolqt(trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml)
    average_ptRes_ml = resolqt(
        true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml
    )

    average_xRes_puppi = resolqt(
        trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi
    )
    average_yRes_puppi = resolqt(
        trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi
    )
    average_ptRes_puppi = resolqt(
        true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi
    )

    # and the resolution magnitudes and the corresponding difference between the puppi and ml predictions
    averageXYmag_Res_puppi = (average_xRes_puppi**2 + average_yRes_puppi**2) ** (0.5)
    averageXYmag_Res_ml = (average_xRes_ml**2 + average_yRes_ml**2) ** (0.5)

    averageXYmag_Res_difference = averageXYmag_Res_puppi - averageXYmag_Res_ml
    averagePt_Res_difference = average_ptRes_puppi - average_ptRes_ml

    # the square root of the number of events in each bin
    nEvents_inBin, _ = np.histogram(binnumber, bins=nbins, range=(1, nbins))
    rootN = np.sqrt(nEvents_inBin)
    nEvents_inBin_phi, _ = np.histogram(binnumberPhi, bins=nbins, range=(1, nbins))
    rootN_phi = np.sqrt(nEvents_inBin_phi)

    # locations of error bars
    binWidth = binnings[1]  # =20
    binCenter = binWidth / 2
    leftOfBinCenter = 0.4 * binWidth  # =8
    rightOfBinCenter = 0.6 * binWidth  # =12

    fig1 = plt.figure(figsize=(15, 12), tight_layout=True)
    fig2 = plt.figure(figsize=(15, 12), tight_layout=True)
    plt.subplots_adjust(wspace=0.2, hspace=0)

    # plot x resolution 20 bins
    ax11 = fig1.add_subplot(2, 2, 1)
    ax11.hlines(
        bin_resolX_ml,
        bin_edges[:-1],
        bin_edges[1:],
        colors="r",
        lw=3,
        label="ML",
        linestyles="solid",
    )
    ax11.hlines(
        bin_resolX_puppi,
        bin_edges[:-1],
        bin_edges[1:],
        colors="g",
        lw=3,
        label="PUPPI",
        linestyles="solid",
    )
    ax11.errorbar(
        bin_edges[:-1] + rightOfBinCenter,
        bin_resolX_ml,
        yerr=bin_resolX_ml / rootN,
        fmt="none",
        color="r",
    )
    ax11.errorbar(
        bin_edges[:-1] + leftOfBinCenter,
        bin_resolX_puppi,
        yerr=bin_resolX_puppi / rootN,
        fmt="none",
        color="g",
    )
    ax11.grid()
    ax11.set_ylabel(r"$\sigma(MET)$ [GeV]", fontsize=19)
    ax11.set_title("MET-x Resolution", fontsize=22)

    # plot y resolutions 20 bins
    ax12 = fig1.add_subplot(2, 2, 2, sharey=ax11)
    ax12.hlines(
        bin_resolY_ml,
        bin_edges[:-1],
        bin_edges[1:],
        colors="r",
        lw=3,
        label="$\\sigma_{ML}$",
        linestyles="solid",
    )
    ax12.hlines(
        bin_resolY_puppi,
        bin_edges[:-1],
        bin_edges[1:],
        colors="g",
        lw=3,
        label="$\\sigma_{PUPPI}$",
        linestyles="solid",
    )
    ax12.errorbar(
        bin_edges[:-1] + rightOfBinCenter,
        bin_resolY_ml,
        yerr=bin_resolY_ml / rootN,
        fmt="none",
        color="r",
    )
    ax12.errorbar(
        bin_edges[:-1] + leftOfBinCenter,
        bin_resolY_puppi,
        yerr=bin_resolY_puppi / rootN,
        fmt="none",
        color="g",
    )
    ax12.legend(loc="upper center", prop={"size": 19})
    ax12.grid()
    ax12.set_title("MET-y Resolution", fontsize=22)

    # plot resolution XY magnitude absolute differences
    ax13 = fig1.add_subplot(2, 2, 3, sharex=ax11)
    ax13.hlines(
        bin_resolXYmagnitude_difference,
        bin_edges[:-1],
        bin_edges[1:],
        lw=5,
        linestyles="solid",
    )
    ax13.axhline(y=0, color="black", linestyle="-")
    ax13.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax13.set_ylabel(r"$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]", fontsize=19)
    ax13.grid()
    ax13.set_title("Absolute Resolution Differences (PUPPI - ML)", fontsize=22)

    # relative differences
    ax14 = fig1.add_subplot(2, 2, 4, sharex=ax12)
    ax14.hlines(
        bin_resolXYmagnitude_difference / truth_means,
        bin_edges[:-1],
        bin_edges[1:],
        lw=5,
        linestyles="solid",
    )
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
        0,
        1.03,
        r"$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{xyDIF}$;  $Mean(\sigma_{xyDIF})=$"
        + f"{round(averageXYmag_Res_difference,3)}",
        fontsize=19,
    )

    # plot pt resolutions 20 bins
    ax21 = fig2.add_subplot(2, 2, 1)
    ax21.hlines(
        bin_resolPt_ml,
        bin_edges[:-1],
        bin_edges[1:],
        colors="r",
        lw=3,
        label="ML",
        linestyles="solid",
    )
    ax21.hlines(
        bin_resolPt_puppi,
        bin_edges[:-1],
        bin_edges[1:],
        colors="g",
        lw=3,
        label="PUPPI",
        linestyles="solid",
    )
    ax21.errorbar(
        bin_edges[:-1] + rightOfBinCenter,
        bin_resolPt_ml,
        yerr=bin_resolPt_ml / rootN,
        fmt="none",
        color="r",
    )
    ax21.errorbar(
        bin_edges[:-1] + leftOfBinCenter,
        bin_resolPt_puppi,
        yerr=bin_resolPt_puppi / rootN,
        fmt="none",
        color="g",
    )
    ax21.set_xlabel("truth met [gev]", fontsize=19)
    ax21.set_ylabel(r"$\sigma(MET)$ [GeV]", fontsize=20)
    ax21.grid()
    ax21.set_title("MET-pt Resolution", fontsize=22)

    # plot phi resolutions 20 bins
    ax22 = fig2.add_subplot(2, 2, 2)
    ax22.hlines(
        bin_resolPhi_ml,
        bin_edgesPhi[:-1],
        bin_edgesPhi[1:],
        colors="r",
        lw=3,
        label="$\\sigma_{ML}$",
        linestyles="solid",
    )
    ax22.hlines(
        bin_resolPhi_puppi,
        bin_edgesPhi[:-1],
        bin_edgesPhi[1:],
        colors="g",
        lw=3,
        label="$\\sigma_{PUPPI}$",
        linestyles="solid",
    )
    ax22.errorbar(
        bin_edgesPhi[:-1] + 0.13,
        bin_resolPhi_ml,
        yerr=bin_resolPhi_ml / rootN_phi,
        fmt="none",
        color="r",
    )
    ax22.errorbar(
        bin_edgesPhi[:-1] + 0.17,
        bin_resolPhi_puppi,
        yerr=bin_resolPhi_puppi / rootN_phi,
        fmt="none",
        color="g",
    )
    ax22.set_ylabel("radian", fontsize=20)
    ax22.set_ylim(0.0, 1.0)
    ax22.grid()
    ax22.set_xlabel(r"$\phi$ angle", fontsize=19)
    ax22.legend(loc="upper center", prop={"size": 19})
    ax22.set_title(r"MET-$\Phi$ Resolution", fontsize=20)

    # plot resolution pt absolute differences
    ax23 = fig2.add_subplot(2, 2, 3, sharex=ax11)
    ax23.hlines(
        bin_resolPt_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles="solid"
    )
    ax23.axhline(y=0, color="black", linestyle="-")
    ax23.set_xlabel("Truth MET [GeV]", fontsize=19)
    ax23.set_ylabel(r"$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]", fontsize=20)
    ax23.grid()
    ax23.set_title("Absolute Resolution Differences", fontsize=22)

    # relative differences
    ax24 = fig2.add_subplot(2, 2, 4, sharex=ax12)
    ax24.hlines(
        bin_resolPt_difference / truth_means,
        bin_edges[:-1],
        bin_edges[1:],
        lw=5,
        linestyles="solid",
    )
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
        0,
        1.03,
        r"$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{DIF}$;  $Mean(\sigma_{DIF})=$"
        + f"{round(averagePt_Res_difference,3)}",
        fontsize=19,
    )

    fig1.savefig(f"{path_out}XY_resolution_plots.png", bbox_inches="tight")
    fig2.savefig(f"{path_out}pt_resolution_plots.png", bbox_inches="tight")


def Make1DHists(
    truth,
    ML,
    PUPPI,
    xmin=0,
    xmax=400,
    nbins=100,
    density=False,
    xname="pt [GeV]",
    yname="A.U.",
    outputname="1ddistribution.png",
):
    """Create 1D histogram plots comparing truth, ML, and PUPPI predictions."""
    plt.figure(figsize=(10, 8))
    plt.hist(
        truth,
        bins=nbins,
        range=(xmin, xmax),
        density=density,
        histtype="step",
        facecolor="k",
        label="Truth",
    )
    plt.hist(
        ML,
        bins=nbins,
        range=(xmin, xmax),
        density=density,
        histtype="step",
        facecolor="r",
        label="ML",
    )
    plt.hist(
        PUPPI,
        bins=nbins,
        range=(xmin, xmax),
        density=density,
        histtype="step",
        facecolor="g",
        label="PUPPI",
    )
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


def MakeEdgeHist(
    edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"
):
    plt.figure(figsize=(10, 8))
    plt.hist(
        edge_feat,
        bins=nbins,
        density=density,
        histtype="step",
        facecolor="k",
        label="Truth",
    )
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close() 


def MET_rel_error_opaque(predict_met, predict_met2, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met

    mask = (rel_err[:] > 3)
    rel_err = rel_err[~mask]

    mean = np.mean(rel_err)
    std = np.std(rel_err)

    entry = rel_err.shape[0]

    rel_err2 = (predict_met2 - gen_met)/gen_met

    mask2 = (rel_err2[:] > 3)
    rel_err2 = rel_err2[~mask2]

    mean = np.mean(rel_err)
    std = np.std(rel_err)

    mean = mean * 1000
    mean = int(mean)
    print(mean, "mean")
    mean = float(mean) / 1000
    std = std * 1000
    std = int(std)
    std = float(std) / 1000

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3., 3., 50+1), label='ML', alpha=0.5, color='red')
    plt.hist(rel_err2, bins=np.linspace(-3., 3., 50+1), label='PUPPI', alpha=0.5, color='green')
    plt.xlabel("relative error (predict - true)/true", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.title('Relative Pt error', size=18, fontweight='bold', loc='right')
    plt.legend()
    plt.savefig(name)
    plt.show(block=False)
    plt.close("all")


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

    X = np.arange(mini, maxi, binning)
    plt.plot(X, X, 'r-')

    plt.xlim(mini, maxi)
    plt.ylim(mini, maxi)
    plt.xlabel('Gen MET mean [GeV]', fontsize=16)
    plt.ylabel('predicted MET mean [GeV]', fontsize=16)
    plt.legend()
    plt.savefig(name)
    plt.show(block=False)
    plt.close("all")


def Phi_abs_error_opaque(puppi_phi, ml_phi, gen_phi, name='Met_res.pdf'):
    puppi_err = (puppi_phi - gen_phi)
    ml_err = (ml_phi - gen_phi)
    plt.figure()
    plt.hist(puppi_err, bins=np.linspace(-3.5, 3.5, 50+1), alpha=0.5, label='puppi')
    plt.hist(ml_err, bins=np.linspace(-3.5, 3.5, 50+1), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Phi error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.show(block=False)
    plt.close("all")


def Pt_abs_error_opaque(puppi_met, ml_met, gen_met, name='Met_res.pdf'):
    puppi_err = (puppi_met - gen_met)
    ml_err = (ml_met - gen_met)
    plt.figure()
    plt.hist(puppi_err, bins=np.linspace(-250, 250, 50+1), alpha=0.5, label='puppi')
    plt.hist(ml_err, bins=np.linspace(-250, 250, 50+1), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Pt error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.show(block=False)
    plt.close("all")


def extract_result(feat_array, targ_array, path, name, mode): #TODO: there should be a better way to handle this than to save np arrays
    np.save(''+path+''+name+'_feature_array_'+mode+'MET', feat_array)
    np.save(''+path+''+name+'_target_array_'+mode+'MET', targ_array)


def histo_2D(predict_pT, gen_pT, min_, max_, name='2D_histo.png'):
    X_hist = np.arange(0, 500, 20)
    Y_hist = X_hist
    plt.plot(X_hist, Y_hist, '-r')
    x_bins = np.linspace(min_, max_, 50)
    y_bins = np.linspace(min_, max_, 50)
    plt.hist2d(gen_pT, predict_pT,  bins=[x_bins, y_bins], cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('gen MET [GeV]')
    plt.ylabel('predicted MET [GeV]')
    plt.savefig(name)
    plt.show(block=False)
    plt.close("all") 


def plot_roc_curve(rates: Dict[str, np.ndarray], auc_scores: Tuple[float, float], 
                   output_dir: str, signal_sample: str, background_sample: str) -> None:
    ml_auc, puppi_auc = auc_scores

    plt.figure(figsize=(8, 6))
    plt.plot(rates['ml_roc'][:, 0], rates['ml_roc'][:, 1],
             label=f'ML ROC, AUC = {ml_auc:.3f}', linewidth=2)
    plt.plot(rates['puppi_roc'][:, 0], rates['puppi_roc'][:, 1],
             label=f'PUPPI ROC, AUC = {puppi_auc:.3f}', linewidth=2, color='red')

    _setup_plot_style()
    plt.xlabel('Signal Efficiency (TPR)', fontsize=16)
    plt.ylabel('Background Efficiency (FPR)', fontsize=16)
    plt.title('ROC Curve: ML vs PUPPI MET', fontsize=18)
    plt.xlim(0., 1.)
    plt.yscale("log")
    plt.ylim(4e-5, 1.1)
    plt.legend(fontsize=14)

    output_path = Path(output_dir) / f'ROC_curve_{signal_sample}_{background_sample}.png'
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def plot_turn_on_curves(
        ml_efficiencies: np.ndarray, 
        ml_err: np.ndarray,
        ml_threshold: float,
        puppi_efficiencies: np.ndarray,
        puppi_err: np.ndarray,
        puppi_threshold: float,
        centers: np.ndarray,
        output_dir: str,
        signal_sample: str,
        background_sample: str) -> None: 
        
    plt.figure(figsize=(8, 6))
    plt.errorbar(centers, ml_efficiencies, yerr= ml_err, linestyle='-',
                label='L1DeepMET', color='blue', linewidth=2, capsize=2)
    plt.errorbar(centers, puppi_efficiencies, yerr= puppi_err, linestyle='-',
                label='PUPPI MET', color='red', linewidth=2, capsize=2)
    _setup_plot_style()
    plt.xlabel('GenMET (GeV)', fontsize=16)
    plt.ylabel('Efficiency at 30 kHz L1 Trigger Rate', fontsize=16)
    plt.title('Turn-On Curves: ML vs PUPPI MET', fontsize=18)
    plt.xlim(0, 600)
    plt.ylim(0, 1.1)
    plt.legend(
        [f'ML MET (Threshold: {ml_threshold:.2f} GeV)',
         f'PUPPI MET (Threshold: {puppi_threshold:.2f} GeV)'],
        fontsize=14
    )

    output_path = Path(output_dir) / f'turn_on_curves_{signal_sample}_{background_sample}.png'
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Turn-on curves saved to {output_path}")


def plot_trigger_rates(rates: Dict[str, np.ndarray], output_dir: str, 
                      signal_sample: str, background_sample: str) -> None:
    """Generate trigger rate plot."""
    """ NB: this is equivalent to ROC curve but BG_eff scaled by trigger rate """

    plt.figure(figsize=(6, 6))
    plt.plot(rates['ml_roc'][:, 0], rates['ml_background_rate'], 'b-', label='ML', markersize=2)
    plt.plot(rates['puppi_roc'][:, 0], rates['puppi_background_rate'], 'r-', label='PUPPI', markersize=2)

    _setup_plot_style()
    plt.xlim(0, 1)
    plt.yscale("log")
    plt.ylim(1, 40)
    plt.xlabel(f'{signal_sample} Signal Efficiency', fontsize=16)
    plt.ylabel(f'MET Trigger Rate (MHz)', fontsize=16)
    plt.title('Trigger Rates: ML vs PUPPI MET', fontsize=18)
    plt.legend(fontsize=14)

    output_path = Path(output_dir) / f'trigger_rates_{signal_sample}v{background_sample}.png'
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Trigger rates plot saved to {output_path}")


def plot_combined_rates(rates: Dict[str, np.ndarray], output_dir: str, 
                       signal_sample: str, background_sample: str, trigger_rate_khz: float) -> None:
    """Generate combined rates plot for ML and PUPPI."""
    plt.figure(figsize=(10, 6))

    ml_bg_rate = rates['ml_background_rate'] * trigger_rate_khz
    puppi_bg_rate = rates['puppi_background_rate'] * trigger_rate_khz

    plt.plot(rates['ml_signal_rate'], ml_bg_rate, 'bo', label='ML', markersize=2)
    plt.plot(rates['puppi_signal_rate'], puppi_bg_rate, 'ro', label='PUPPI', markersize=2)

    _setup_plot_style()
    plt.yscale("log")
    plt.xlabel(f'{signal_sample} Efficiency')
    plt.ylabel(f'{background_sample} (kHz)')
    plt.title('Combined Rates: ML vs PUPPI MET')
    plt.legend()

    output_path = Path(output_dir) / 'combined_rates.png'
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Combined rates plot saved to {output_path}")


def _setup_plot_style() -> None:
    """Apply consistent style to plots."""
    plt.grid(True, color='gray', alpha=0.5, linestyle='--')
    plt.tight_layout() 


def create_histogram_plots(features, output_dir, bins=100):
    """
    Create histogram plots for features with log scale on y-axis
    
    Args:
        features: Dictionary with feature names as keys and arrays as values
        output_dir: Directory to save plots
        bins: Number of bins for histograms
    """
    setup_cms_style()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for feature_name, data in features.items():
        if len(data) == 0:
            print(f"Warning: No data for {feature_name}")
            continue
            
        plt.figure(figsize=(10, 8))
        
        # Remove extreme outliers for better visualization
        q1, q99 = np.percentile(data, [1, 99])
        filtered_data = data[(data >= q1) & (data <= q99)]
        
        counts, bin_edges, patches = plt.hist(
            filtered_data, 
            bins=bins, 
            histtype='step', 
            linewidth=2,
            label=f'{feature_name} (n={len(filtered_data):,})',
            alpha=0.8
        )
        
        plt.yscale('log')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
        plt.title(f'{feature_name} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        stats_text = f'Mean: {np.mean(filtered_data):.3f}\n'
        stats_text += f'Std: {np.std(filtered_data):.3f}\n'
        stats_text += f'Median: {np.median(filtered_data):.3f}\n'
        stats_text += f'Min: {np.min(filtered_data):.3f}\n'
        stats_text += f'Max: {np.max(filtered_data):.3f}'
        
        plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_file = os.path.join(output_dir, f'{feature_name}_histogram.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_file}")
    
    if len(features) > 1:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (feature_name, data) in enumerate(features.items()):
            if len(data) == 0:
                continue
                
            # Remove extreme outliers for better visualization
            q1, q99 = np.percentile(data, [1, 99])
            filtered_data = data[(data >= q1) & (data <= q99)]
            
            plt.hist(
                filtered_data, 
                bins=bins, 
                histtype='step', 
                linewidth=2,
                label=f'{feature_name} (n={len(filtered_data):,})',
                alpha=0.8,
                color=colors[i % len(colors)]
            )
        
        plt.yscale('log')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')
        plt.title('Feature Distributions Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'combined_features_histogram.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_file}")


def create_correlation_plot(features, output_dir):
    """
    Create correlation plot between features if multiple features are present
    
    Args:
        features: Dictionary with feature names as keys and arrays as values
        output_dir: Directory to save plots
    """
    if len(features) < 2:
        return
        
    setup_cms_style()
    
    feature_names = list(features.keys())
    feature_data = np.column_stack([features[name] for name in feature_names])
    
    correlation_matrix = np.corrcoef(feature_data.T)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10)
    
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'feature_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


def analyze_outliers(features, output_dir):
    """
    Analyze and report outliers in the features
    
    Args:
        features: Dictionary with feature names as keys and arrays as values
        output_dir: Directory to save analysis results
    """
    analysis_file = os.path.join(output_dir, 'outlier_analysis.txt')
    
    with open(analysis_file, 'w') as f:
        f.write("Feature Outlier Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        for feature_name, data in features.items():
            if len(data) == 0:
                continue
                
            f.write(f"{feature_name}:\n")
            f.write(f"  Total values: {len(data):,}\n")
            f.write(f"  Mean: {np.mean(data):.6f}\n")
            f.write(f"  Std: {np.std(data):.6f}\n")
            f.write(f"  Min: {np.min(data):.6f}\n")
            f.write(f"  Max: {np.max(data):.6f}\n")
            
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            f.write("  Percentiles:\n")
            for p in percentiles:
                f.write(f"    {p:2d}%: {np.percentile(data, p):.6f}\n")
            
            zero_count = np.sum(data == 0)
            neg999_count = np.sum(data == -999)
            inf_count = np.sum(~np.isfinite(data))
            
            f.write(f"  Zero values: {zero_count:,} ({100*zero_count/len(data):.2f}%)\n")
            f.write(f"  -999 values: {neg999_count:,} ({100*neg999_count/len(data):.2f}%)\n")
            f.write(f"  Non-finite values: {inf_count:,} ({100*inf_count/len(data):.2f}%)\n")
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            outliers_3sigma = np.sum(np.abs(data - mean_val) > 3 * std_val)
            f.write(f"  Values beyond 3σ: {outliers_3sigma:,} ({100*outliers_3sigma/len(data):.2f}%)\n")
            
            f.write("\n")
    
    print(f"Saved outlier analysis to {analysis_file}") 



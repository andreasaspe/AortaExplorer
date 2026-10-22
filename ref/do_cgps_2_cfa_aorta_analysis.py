import datetime
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
from networkx.classes import neighbors
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

def write_message_to_log_file(base_dir, message, scan_id=None, level="warning"):
    log_file = os.path.dirname(base_dir) + f"/cfa_analysis_log.txt"
    log_csv = os.path.dirname(base_dir) + f"/cfa_analysis_log.csv"
    if not os.path.isdir(os.path.dirname(log_file)):
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)

    now_date = datetime.strftime(datetime.now(), "%d-%m-%Y-%H-%M-%S")
    with open(log_file, "a") as file:
        file.write(f"{now_date}: {message}\n")

    if scan_id is not None:
        with open(log_csv, "a") as file:
            file.write(f"{level},{scan_id},{message},{now_date}\n")


def analyze_cgps_2_aorta_data_max_diam_diff():
    print("Analyzing")
    data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta/aorta-combined_statistics-28-04-2025-21-41-24.csv"
    out_dir = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-analysis-29-04-2025/"
    dataset = "CGPS-2"

    # Plot using pandas
    df = pd.read_csv(data_in)
    # print(df.columns)
    # print(df.head())
    # print(df.describe())
    # print(df.info())
    # print(df.dtypes)
    # print(df.isnull().sum())
    # print(df.isna().sum())
    # print(df.shape)
    # df.fillna(100, inplace=True)

    x = df["infrarenal_cl_max_diameter"]
    y = df["infrarenal_ts_original_cl_max_diameter"]

    n_samp_ae = 0
    n_samp = 0
    for i in range(len(x)):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            n_samp += 1
        if not np.isnan(x[i]):
            n_samp_ae += 1

    df["infrarenal_cl_max_diameter"].plot(kind = 'hist', bins=50, title=f'Infrarenal Max Diameter Histogram for {dataset} for {n_samp_ae}')
    plt.savefig(f"{out_dir}InfrarenalMaxDiamHist.png")
    plt.show()

    # df.plot(kind = 'scatter', x='infrarenal_cl_max_diameter', y='infrarenal_ts_original_cl_max_diameter', title='Infrarenal Max Diameters')
    # plt.show()

    plt.scatter(x=df["infrarenal_cl_max_diameter"], y=df["infrarenal_ts_original_cl_max_diameter"])
    plt.plot([10, 50], [10, 50], 'r-', lw=2)
    plt.xlabel("Aorta Explorer")
    plt.ylabel("TotalSegmentator")
    plt.title(f"Infrarenal Max Diameters for {dataset} n={n_samp}")
    plt.savefig(f"{out_dir}InfrarenalMaxDiam_TS_vs_AE.png")
    # plt.savefig(f"C:/data/DTUImageMeasures/DTU-CGPS-2/{ati_type}_tortuosity_vs_age.png")
    plt.show()

    # sns.regplot(x=df["infrarenal_cl_max_diameter"], y=df["infrarenal_ts_original_cl_max_diameter"], ci=False, line_kws={'color':'red'})
    # plt.xlabel("infrarenal_cl_max_diameter")
    # plt.ylabel("infrarenal_ts_original_cl_max_diameter")
    # # plt.title("Descending Tortuosity vs Age")
    # plt.title(f"Infrarenal Max Diameters'")
    # # plt.savefig(f"C:/data/DTUImageMeasures/DTU-CGPS-2/{ati_type}_tortuosity_vs_age.png")
    # plt.show()

    diff = df["infrarenal_ts_original_cl_max_diameter"] - df["infrarenal_cl_max_diameter"]
    diff.plot(kind = 'hist', bins=50, title=f'Difference Histogram {dataset} n={n_samp}')
    plt.savefig(f"{out_dir}InfrarenalMaxDiam_TS_vs_AE_hist.png")
    plt.show()

    diff_p = (df["infrarenal_ts_original_cl_max_diameter"] - df["infrarenal_cl_max_diameter"]) / df["infrarenal_cl_max_diameter"] * 100
    diff_p.plot(kind = 'hist', bins=50, title=f'Difference Percentage Histogram {dataset} n={n_samp}')
    plt.savefig(f"{out_dir}InfrarenalMaxDiam_TS_vs_AE_hist_percent.png")
    plt.show()
    diff_p.plot(kind = 'box', title='Difference Percentage Boxplot')
    plt.show()


def analyze_cgps_2_aorta_tortuosity():
    print("Analyzing")
    data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta/aorta-combined_statistics-28-04-2025-21-41-24.csv"
    # age_in = "C:/data/DTUImageMeasures/DTU-CGPS-2/ages.csv"

    # df_age = pd.read_csv(age_in)
    # df_age["Age"].plot(kind = 'hist', bins=50, title='Patient Age')
    # plt.show()

    df = pd.read_csv(data_in)
    # df["ib_arch_aortic_tortuosity_index"].plot(kind = 'hist', bins=50, title='Ib-Aortic Arch Tortuosity Index Histogram')
    # plt.show()

    # ages = df_age["Age"].values
    # pseudo_id_age = df_age["PseudoId"].values

    ati_type = "infrarenal_cl_max_diameter"
    # ati_type = "Annulus"
    # ati_type = "ib_arch"
    # ati_type = "Annulus_geom"
    # ati_type = "Annulus_cl_length"
    # ati_type = "calcification_volume"
    # ati_type = "Infrarenal"
    # descending = True
    if ati_type == "descending":
        tortuosity = df["descending_aortic_tortuosity_index"].values
    elif ati_type == "ib_arch":
        tortuosity = df["ib_arch_aortic_tortuosity_index"].values
    elif ati_type == "Annulus":
        tortuosity = df["annulus_aortic_tortuosity_index"].values
    elif ati_type == "Infrarenal":
        tortuosity = df["infrarenal_aortic_tortuosity_index"].values
    elif ati_type == "Annulus_geom":
        tortuosity = df["annulus_geometric_length"].values
    elif ati_type == "Annulus_cl_length":
        tortuosity = df["annulus_aortic_length"].values
    elif ati_type == "calcification_volume":
        tortuosity = df["calcification_volume"].values
    elif ati_type == "infrarenal_cl_max_diameter":
        tortuosity = df["infrarenal_cl_max_diameter"].values
    # scan_id = df["scan_name"].values

    ages = df["patient_age"]

    # age_list = []
    # tort_list = []
    # for i in range(len(tortuosity)):
    #     tor = tortuosity[i]
    #     if not np.isnan(tor):
    #         s_id = scan_id[i]
    #         for j in range(len(pseudo_id_age)):
    #             p_id = pseudo_id_age[j] + "_"
    #             p_id = p_id.replace("CGPS", "CFA")
    #             if p_id in s_id:
    #                 age = ages[j]
    #                 age_list.append(age)
    #                 tort_list.append(tor)
    #                 break

    # print(f"Found {len(age_list)} patients with tortuosity and age")

    # plt.scatter(age_list, tort_list)
    # plt.show()

    sns.regplot(x=ages, y=tortuosity, ci=False, line_kws={'color':'red'})
    plt.xlabel("Age")
    plt.ylabel("Tortuosity")
    # plt.title("Descending Tortuosity vs Age")
    plt.title(f"{ati_type} Tortuosity vs Age")
    # plt.savefig(f"C:/data/DTUImageMeasures/DTU-CGPS-2/{ati_type}_tortuosity_vs_age.png")
    plt.show()

    ages = np.asarray(ages)
    filter_age = ages > 65

    ati_above_65 = tortuosity[filter_age]
    ati_above_65 = ati_above_65[~np.isnan(ati_above_65)]

    ati_below_65 = tortuosity[~filter_age]
    ati_below_65 = ati_below_65[~np.isnan(ati_below_65)]

    t_stat, p_val = stats.ttest_ind(ati_above_65, ati_below_65)
    print("t-statistic = " + str(t_stat))
    print("p-value = " + str(p_val))

    # plt.boxplot(ati_above_65)
    # plt.show()
    #
    # plt.boxplot(ati_below_65)
    # plt.show()

    labels = ["Below 66", "Above 65"]

    combined_atr = [ati_below_65, ati_above_65]
    plt.boxplot(combined_atr)
    # plt.xlabel(labels)
    plt.show()


def analyze_dataset_distributions():
    print("Analyzing")
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta/aorta-combined_statistics-28-04-2025-21-41-24.csv"
    # dataset = "CGPS-2"
    # out_dir = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-analysis-29-04-2025/"

    data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-1/Aorta/aorta-combined_statistics-01-05-2025-12-47-04.csv"

    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-1/Aorta/aorta-combined_statistics-29-04-2025-22-29-59.csv"
    dataset = "CGPS-1"
    out_dir = "/data/Data/DTUImageMeasures/DTU-CGPS-1/Aorta-analysis-01-05-2025/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_in)

    # plot_type = {"series": "annulus_aortic_tortuosity_index", "title": "Annulus ATI"}
    # plot_type = {"series": "ib_arch_aortic_tortuosity_index", "title": "Aortic arch to IB ATI"}
    # plot_type = {"series": "calcification_volume", "title": "Total Calcification Volume (mm³)"}
    # plot_type = {"series": "descending_aortic_tortuosity_index", "title": "Descending ATI"}
    # plot_type = {"series": "infrarenal_aortic_tortuosity_index", "title": "Infrarenal ATI"}
    # plot_type = {"series": "annulus_geometric_length", "title": "Annulus distance to IB (mm)"}
    # plot_type = {"series": "annulus_aortic_length", "title": "Annulus centerline distance to IB (mm)"}
    # plot_type = {"series": "infrarenal_aortic_length", "title": "Infrarenal centerline distance to IB (mm)"}
    # plot_type = {"series": "ib_arch_geometric_length", "title": "Aortic arch to IB distance (mm)"}
    # plot_type = {"series": "calcification_ratio", "title": "Calcification ratio (%)"}
    # plot_type = {"series": "ascending_cl_max_diameter", "title": "ascending_cl_max_diameter (mm)"}
    plot_type = {"series": "descending_cl_max_diameter", "title": "descending_cl_max_diameter (mm)"}


    ages = df["patient_age"]
    y = df[plot_type["series"]]
    n_samp = 0
    for i in range(len(y)):
        if not np.isnan(y[i]):
            n_samp += 1

    title = plot_type["title"]
    sns.regplot(x=ages, y=y, ci=False, line_kws={'color':'red'})
    plt.xlabel("Age")
    plt.ylabel(title)
    # plt.title("Descending Tortuosity vs Age")
    plt.title(f"{title} vs Age\n{dataset} n={n_samp}")
    plt.savefig(f"{out_dir}{title}_vs_age_for_{dataset}.png")
    plt.show()

    filter_age = ages > 65

    val_above_65 = y[filter_age]
    val_above_65 = val_above_65[~np.isnan(val_above_65)]

    val_below_65 = y[~filter_age]
    val_below_65 = val_below_65[~np.isnan(val_below_65)]

    t_stat, p_val = stats.ttest_ind(val_above_65, val_below_65)
    print(f"t-statistic = {t_stat:.2f}")
    print(f"p-value = {p_val:.5f}")

    # plt.boxplot(ati_above_65)
    # plt.show()
    #
    # plt.boxplot(ati_below_65)
    # plt.show()

    labels = ["Below 66", "Above 65"]

    combined_atr = [val_below_65, val_above_65]
    plt.boxplot(combined_atr)
    plt.title(f"{title}\nBelow 65 years vs above 65 years\n{dataset} n={n_samp} p={p_val:.5f}")
    plt.savefig(f"{out_dir}{title}_65_test_box_{dataset}.png")
    # plt.xlabel(labels)
    plt.show()

def analyze_cgps_2_cfa_aorta():
    print("Analyzing")

    # ser_name = "CFA-1"
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-1/aorta-statistics-13-05-2025-22-12-02.csv"
    # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-1/whole_heart-statistics-12-05-2025-22-18-03.csv"

    # ser_name = "CFA-2"
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-2/aorta-statistics-13-05-2025-22-18-11.csv"
    # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-2/whole_heart-statistics-12-05-2025-22-14-33.csv"

    # ser_name = "CFA-3"
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-3/aorta-statistics-13-05-2025-22-24-48.csv"
    # # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-3/whole_heart-statistics-12-05-2025-21-33-17.csv"



    # ser_name = "CFA-4"
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-4/aorta-statistics-12-05-2025-22-30-55.csv"
    # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-4/whole_heart-statistics-12-05-2025-22-28-21.csv"

    # ser_name = "CFA-5"
    # data_in = "/data/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-5/aorta-statistics-13-05-2025-22-08-31.csv"
    # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-5/whole_heart-statistics-13-05-2025-21-25-26.csv"

    ser_name = "CFA-6"
    data_in = "/storage/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-6/aorta-statistics-13-05-2025-22-35-15.csv"
    # date_in_wh = "/data/Data/DTUImageMeasures/DTU-CGPS-2/WholeHeart-CFA-3/whole_heart-statistics-12-05-2025-21-33-17.csv"


    out_dir = "/storage/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-analysis-12-05-2025/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_in)
    # df_wh = pd.read_csv(date_in_wh)

    asc_a = df["ascending_area"] / 100
    desc_a = df["descending_area"] / 100
    sino_a = df["sinotubular_junction_area"] / 100
    # la_vol = df_wh["heartchambers_highres_heart_atrium_left_tot_vol"] / 1000
    # lv_vol = df_wh["heartchambers_highres_heart_ventricle_left_tot_vol"] /1000
    la_vol = df["vol_heart_left_atrium"] / 1000
    lv_vol = df["vol_heart_left_ventricle"] / 1000
    asc_vol = df["vol_ascending_aorta"] / 1000
    desc_vol = df["vol_descending_aorta"] / 1000

    # plt.plot(asc_a, label="Ascending area")
    # plt.plot(desc_a, label="Descending area")
    # plt.plot(la_vol, label="Left atrium volume")
    # plt.plot(lv_vol, label="Left ventricle volume")
    # plt.show()

    systolic_bp = 140
    diastolic_bp = 80
    bp_dif = systolic_bp - diastolic_bp
    min_asc_a = np.min(asc_a)
    max_asc_a = np.max(asc_a)
    meas_1 = (max_asc_a - min_asc_a) / (min_asc_a * bp_dif) * 1000
    max_lv = np.max(lv_vol)
    min_lv = np.min(lv_vol)
    ef = (max_lv - min_lv) / max_lv * 100
    min_asc_vol = np.min(asc_vol)
    max_asc_vol = np.max(asc_vol)
    dist_vol = (max_asc_vol - min_asc_vol) / (min_asc_vol * bp_dif) * 1000


    title = (f"Ascending A min={min_asc_a:.2f} (cm²) max={max_asc_a:.2f} (cm²)\nBP systolic={systolic_bp} diastolic={diastolic_bp}\n"
             f"Ejection Fraction={ef:.2f}\nAortic Distensibility={meas_1:.2f} 10⁻3 mmHg^-1\n"
             f"Aortic Distensibility from Volume={dist_vol:.2f} 10⁻³ mmHg^-1\n")

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 10)

    color = 'tab:red'
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Area (cm²)', color=color)
    ax1.plot(asc_a,  label="Ascending area (cm²)", color="red")
    ax1.plot(desc_a,label="Descending area (cm²)", color="orange")
    ax1.plot(sino_a,label="Sinotubular area (cm²)", color="yellow")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Volume (cm³)', color=color)  # we already handled the x-label with ax1
    ax2.plot(la_vol, label="Left atrium volume (cm³)", color="blue")
    ax2.plot(lv_vol, label="Left ventricle volume (cm³)", color="cyan")
    ax2.plot(asc_vol, label="Ascending aorta volume (cm³)", color="purple")
    ax2.plot(desc_vol, label="Descending aorta volume (cm³)", color="darkblue")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper right")
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{out_dir}CGPS-2-{ser_name}.png")
    plt.show()


def analyze_nstemi_cfa_aorta_patient(patient_id):
    # patient_id = "NSTEMI_0003_"

    cfa_list = "/storage/Data/DTU-NSTEMI/DTU-NSTEMI/Filelists/NSTEMI_CFA_pecentage.csv"
    analysis_results_in = "/data/Data/DTUImageMeasures/NSTEMI/aorta-statistics-09-08-2025-15-00-36.csv"
    out_dir = "/data/Data/DTUImageMeasures/NSTEMI/figures/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Read percentage, filename from csv file using pandas
    colnames = ["percent", "filename"]
    df = pd.read_csv(cfa_list, names=colnames, header=None)

    percents = []
    found_scans = []
    for row in df.itertuples():
        percent = row.percent
        filename = row.filename

        if patient_id not in filename:
            continue

        filename = filename.replace(".nii.gz", "")
        percents.append(percent)
        found_scans.append(filename)

    print(f"Found {len(percents)} scans for {patient_id}")

    results = pd.read_csv(analysis_results_in)

    # Create new dataframe with the rows where the filename is in the found_scans list
    filtered_results = results[results["scan_name"].isin(found_scans)]
    print(f"Found {len(filtered_results)} scans in the analysis results for {patient_id}")

    n_results = len(filtered_results)
    if n_results == 0:
        print(f"No results found for patient {patient_id} in the analysis results.")
        return

    # Add the percentage column to the filtered results
    # filtered_results["cfa_percentage"] = np.nan
    for i, row in filtered_results.iterrows():
        scan_name = row["scan_name"]
        for j, scan in enumerate(found_scans):
            if scan in scan_name:
                filtered_results.at[i, "cfa_percentage"] = percents[j]
                break

    # Sort the filtered results by cfa_percentage
    filtered_results = filtered_results.sort_values(by="cfa_percentage")

    filtered_desc = filtered_results["SeriesDescription"].values
    print(f"Filtered descriptions: {filtered_desc}")


    perc = filtered_results["cfa_percentage"].values
    asc_a = filtered_results["ascending_area"] / 100
    # desc_a = filtered_results["descending_area"] / 100
    dist_a = filtered_results["distensability_segment_avg_slice_info_area"] / 100
    # la_vol = df_wh["heartchambers_highres_heart_atrium_left_tot_vol"] / 1000
    # lv_vol = df_wh["heartchambers_highres_heart_ventricle_left_tot_vol"] /1000
    la_vol = filtered_results["vol_heart_left_atrium"] / 1000
    lv_vol = filtered_results["vol_heart_left_ventricle"] / 1000
    # asc_vol = filtered_results["vol_ascending_aorta"] / 1000
    # desc_vol = filtered_results["vol_descending_aorta"] / 1000

    # plt.plot(asc_a, label="Ascending area")
    # plt.plot(desc_a, label="Descending area")
    # plt.plot(la_vol, label="Left atrium volume")
    # plt.plot(lv_vol, label="Left ventricle volume")
    # plt.show()

    systolic_bp = 140
    diastolic_bp = 80
    bp_dif = systolic_bp - diastolic_bp
    min_asc_a = np.min(asc_a)
    max_asc_a = np.max(asc_a)
    meas_1 = (max_asc_a - min_asc_a) / (min_asc_a * bp_dif) * 1000
    max_lv = np.max(lv_vol)
    min_lv = np.min(lv_vol)
    ef = (max_lv - min_lv) / max_lv * 100

    # min_asc_vol = np.min(asc_vol)
    # max_asc_vol = np.max(asc_vol)
    # dist_vol = (max_asc_vol - min_asc_vol) / (min_asc_vol * bp_dif) * 1000
    min_dist_area = np.min(dist_a)
    max_dist_area = np.max(dist_a)
    dist_dist_area = (max_dist_area - min_dist_area) / (min_dist_area * bp_dif) * 1000

    # Remove last underscore from patient_id
    patient_id = patient_id.rstrip("_")

    title = (f"Patient {patient_id} with {n_results} computed scans\n"
             f"Ascending A min={min_asc_a:.2f} (cm²) max={max_asc_a:.2f} (cm²)\n"
             f"Fixed point A min={min_dist_area:.2f} (cm²) max={max_dist_area:.2f} (cm²)\n"
             f"BP systolic={systolic_bp} diastolic={diastolic_bp}\n"
             f"Ejection Fraction={ef:.2f}\n"
             f"Aortic Distensibility from max area={meas_1:.2f} 10⁻3 mmHg^-1\n"
             f"Aortic Distensibility from fixed place={dist_dist_area:.2f} 10⁻³ mmHg^-1\n")

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 10)

    color = 'tab:red'
    ax1.set_xlabel('CFA Percentage')
    ax1.set_ylabel('Area (cm²)', color=color)
    ax1.plot(perc, asc_a,  label="Ascending area (cm²)", color="red")
    # ax1.plot(desc_a,label="Descending area (cm²)", color="orange")
    ax1.plot(perc, dist_a,label="Distensability area (cm²)", color="yellow")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Volume (cm³)', color=color)  # we already handled the x-label with ax1
    ax2.plot(perc, la_vol, label="Left atrium volume (cm³)", color="blue")
    ax2.plot(perc, lv_vol, label="Left ventricle volume (cm³)", color="cyan")
    # ax2.plot(asc_vol, label="Ascending aorta volume (cm³)", color="purple")
    # ax2.plot(desc_vol, label="Descending aorta volume (cm³)", color="darkblue")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper right")
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f"{out_dir}{patient_id}_Distensability.png")
    # plt.show()


def align_all_profiles_using_cross_correlation(base_dir, patient_id, series_list, sinus_of_valsava_idx=None):
    """
    Aligns a list of series using cross-correlation.
    Each series is expected to have 'dist' and 'area' keys.
    Returns a list of aligned series.
    Only uses the 40 samples in the middle of the series for alignment.
    The first series is used as the reference.
    The rest of the series are shifted based on the cross-correlation with the first series.
    40 samples are used to find the best shift.
    The distance is shifted based on the best shift found.
    """
    debug = False

    ref_series = series_list[0]

    # Find maximum area and the idx of the maximum area in the reference series
    # max_area = np.max(ref_series['area'])
    max_area_idx = sinus_of_valsava_idx

    # max_area_idx = np.argmax(ref_series['area'])
    #
    # if max_area_idx < 10 or max_area_idx > len(ref_series['area']) - 10:
    #     print("Maximum area index is too close to the edges of the series, using middle.")
    #     max_area_idx = len(ref_series['area']) // 2  # Fallback to the middle of the series
    #
    # print(f"Maximum area: {max_area} at index {max_area_idx} sinus of valsava idx: {sinus_of_valsava_idx}")

    ref_idx_min = max(max_area_idx - 50, 10)
    ref_idx_max = min(max_area_idx + 10, len(ref_series['area']) - 10)

    if debug:
        # plot the reference series area
        plt.plot(ref_series['area'][ref_idx_min:ref_idx_max], label=f"Reference Series {ref_series['scan_name']}")
        plt.xlabel("Sample Index")
        plt.ylabel("Area")
        plt.title(f"Reference Series Area for {ref_series['scan_name']}")
        plt.legend()
        # plt.savefig(f"{ref_series['scan_name']}_reference_area.png")
        plt.show()


    min_accepted_corr = 0.80

    aligned_series = []
    for series in series_list:
        opt_shift = -np.inf
        opt_corr = -np.inf

        for shift in range(-20, 21):
            # Shift the area of the current series
            shifted_area = np.roll(series['area'], shift)
            # Calculate the correlation with the reference series
            # use a normalized correlation
            # Only use the part of the series that is within the reference indices
            if len(ref_series['area'][ref_idx_min:ref_idx_max]) != len(shifted_area[ref_idx_min:ref_idx_max]):
                print(f"Somehow the lengths of the series do not match: {len(ref_series['area'][ref_idx_min:ref_idx_max])} != {len(shifted_area[ref_idx_min:ref_idx_max])}")
                return None

            corr = np.corrcoef(ref_series['area'][ref_idx_min:ref_idx_max], shifted_area[ref_idx_min:ref_idx_max])[0, 1]

            if corr > opt_corr:
                opt_corr = corr
                opt_shift = shift

        if opt_corr > min_accepted_corr:
            new_series = {
                'dist': np.roll(series['dist'], opt_shift),
                'area': np.roll(series['area'], opt_shift),
                'out_of_scan': series['out_of_scan'],
                'percent': series['percent'],
                'scan_name': series['scan_name']
            }
            aligned_series.append(new_series)
        else:
            write_message_to_log_file(base_dir, f"Optimal correlation {opt_corr} less than {min_accepted_corr} "
                                                f"found for patient {patient_id}", patient_id,
                                      "warning")
            print(f"Optimal correlation {opt_corr} less than {min_accepted_corr}")

        if debug:
            print(f"Shift for {series['scan_name']}: {opt_shift} with correlation {opt_corr:.4f}")

    return aligned_series


def compute_distensibility_from_series(meas_series):
    systolic_bp = 140
    diastolic_bp = 80
    bp_dif = systolic_bp - diastolic_bp
    min_a = np.min(meas_series)
    max_a = np.max(meas_series)
    if min_a != 0:
        meas_1 = (max_a - min_a) / (min_a * bp_dif) * 1000
    else:
        meas_1 = 0
    return meas_1


def identify_series_outliers(base_dir, series, patient_id, msg):
    """
    Identifies outliers in the series based on the area values.
    Outliers are defined as those with an area value more than 1.5 times the interquartile range (IQR) above the third quartile.
    Returns a list of outlier series names.
    """

    # Detect oultliers as those that are more or less than 3 standard deviations from the mean
    mean = np.mean(series)
    std_dev = np.std(series)
    upper_bound = mean + 3 * std_dev
    lower_bound = mean - 3 * std_dev
    if np.any(series > upper_bound):
        write_message_to_log_file(base_dir, f"Outliers found for patient {patient_id} (high)", patient_id, "warning")
        print(f"Outliers found for patient {patient_id} (high)")
        msg += f"High Outliers found\n"
    if np.any(series < lower_bound):
        write_message_to_log_file(base_dir, f"Outliers found for patient {patient_id} (low)", patient_id, "warning")
        print(f"Outliers found for patient {patient_id} (low)")
        msg += f"Low outliers found\n"


    # Compare each element with its 5 neighbours and see if is too different
    # for i in range(len(series)):
    #     min_idx = i - 2
    #     max_idx = i + 3
    #
    #     neighbors = []
    #     for j in range(min_idx, max_idx):
    #         if j >= len(series):
    #             j -= len(series)
    #         if j != i:
    #             neighbors.append(series[j])
    #
    #     median = np.median(neighbors)
    #     mean = np.mean(neighbors)
    #     std = np.std(neighbors)
    #     val = series[i]
    #     if val < mean - 3 * std:
    #         write_message_to_log_file(base_dir, f"Neighbour outliers found for patient {patient_id} (low)", patient_id, "warning")
    #         print(f"Neighbour outliers found for patient {patient_id} (low) at idx {i} with value {val:.2f} and neighbor mean {mean:.2f} std {std:.2f}")
    #         msg += f"Neighbour low outliers found at idx {i} with value {val:.2f}\n"
    #     if val > mean + 3 * std:
    #         write_message_to_log_file(base_dir, f"Neighbour outliers found for patient {patient_id} (high)", patient_id, "warning")
    #         print(f"Neighbour outliers found for patient {patient_id} (high) at idx {i} with value {val:.2f} and neighbor mean {mean:.2f} std {std:.2f}")
    #         msg += f"Neighbour high outliers found at idx {i} with value {val:.2f}\n"

    new_series = []
    for i in range(len(series)):
        min_idx = i - 1
        max_idx = i + 1
        if max_idx >= len(series):
            max_idx -= len(series)

        left_n = series[min_idx]
        right_n = series[max_idx]

        dif_left = (series[i] - left_n)
        dif_right = (series[i] - right_n)
        sum_dif = dif_left + dif_right
        dif_dif = abs(dif_left - dif_right)

        if (dif_left > 40 and dif_right > 40) or (sum_dif > 80 and dif_dif < 20):
            write_message_to_log_file(base_dir, f"Large jump outliers found for patient {patient_id} (high)", patient_id, "warning")
            print(f"Large jump outliers found for patient {patient_id} (high) at idx {i} with value {series[i]:.2f} dif left {dif_left:.2f} dif right {dif_right:.2f}")
            msg += (f"Large jump high outliers found at idx {i} with value {series[i]:.2f} "
                    f"dif left {dif_left:.2f} dif right {dif_right:.2f}\n")
            new_series.append(np.mean([left_n, right_n]))
        elif (dif_left < -40 and dif_right < -40) or (sum_dif < -80 and dif_dif < 20):
            write_message_to_log_file(base_dir, f"Large jump outliers found for patient {patient_id} (low)", patient_id, "warning")
            print(f"Large jump outliers found for patient {patient_id} (low) at idx {i} with value {series[i]:.2f} dif left {dif_left:.2f} dif right {dif_right:.2f}")
            msg += (f"Large jump low outliers found at idx {i} with value {series[i]:.2f} "
                    f"dif left {dif_left:.2f} dif right {dif_right:.2f}\n")
            new_series.append(np.mean([left_n, right_n]))
        else:
            new_series.append(series[i])


    return msg, new_series
    #
    # q1 = np.percentile(series, 25)
    # q3 = np.percentile(series, 75)
    # iqr = q3 - q1
    # upper_bound = q3 + 1.5 * iqr
    # lower_bound = q1 - 1.5 * iqr
    #
    # # outliers = []
    # if np.any(series > upper_bound):
    #     write_message_to_log_file(base_dir, f"Outliers found for patient {patient_id} (high)", patient_id, "warning")
    #     print(f"Outliers found for patient {patient_id} (high)")
    # if np.any(series < lower_bound):
    #     write_message_to_log_file(base_dir, f"Outliers found for patient {patient_id} (low)", patient_id, "warning")
    #     print(f"Outliers found for patient {patient_id} (low)")

    # return outliers



def analyze_nstemi_cfa_aorta_area_profiles(patient_id, result_f=None):
    cfa_list = "/storage/Data/DTU-NSTEMI/Filelists/NSTEMI_CFA_percentage.csv"
    # analysis_results_in = "/data/Data/DTUImageMeasures/NSTEMI/aorta-statistics-09-08-2025-15-00-36.csv"
    out_dir = "/storage/Data/DTU-NSTEMI/DTUImageMeasures/NSTEMI/figures/"
    base_dir = "/storage/Data/DTU-NSTEMI/DTUImageMeasures/NSTEMI/"

    # cfa_list = "/storage/Data/DTUImageMeasures/DTU-CGPS-2/CGPS-2_0049_20201021_260_20_DUBLETS_3_cfa.csv"
    # # analysis_results_in = "/data/Data/DTUImageMeasures/NSTEMI/aorta-statistics-09-08-2025-15-00-36.csv"
    # out_dir = "/storage/Data/DTUImageMeasures/DTU-CGPS-2/figures/"
    # base_dir = "/storage/Data/DTUImageMeasures/DTU-CGPS-2/Aorta-CFA-1/"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    debug = False

    # Read percentage, filename from csv file using pandas
    colnames = ["percent", "filename"]
    df = pd.read_csv(cfa_list, names=colnames, header=None)

    fig_msg = ""

    percents = []
    found_scans = []
    for row in df.itertuples():
        percent = row.percent
        filename = row.filename

        if patient_id not in filename:
            continue

        filename = filename.replace(".nii.gz", "")
        percents.append(percent)
        found_scans.append(filename)

    # print(f"Found {len(percents)} scans for {patient_id}")
    if len(percents) < 5:
        print(f"Less than five scans found for patient {patient_id} exiting.")
        write_message_to_log_file(base_dir, f"Less than five scans found for patient {patient_id} exiting.",
                                  patient_id, "error")
        return

    if len(percents) > 20:
        print(f"More than twenty scans found for patient {patient_id} exiting.")
        write_message_to_log_file(base_dir, f"More than twenty scans found for patient {patient_id} exiting.",
                                  patient_id, "error")
        return

    all_series = []
    for i, scan in enumerate(found_scans):
        # print(f"Processing scan {i + 1}/{len(found_scans)}: {scan}")

        in_series_name = f"{base_dir}{scan}/centerlines/straight_cl_sampling_labelmap_annulus.csv"

        if not os.path.exists(in_series_name):
            # print(f"File {in_series_name} does not exist, skipping.")
            continue

        colnames = ["dist", "area", "out_of_scan"]
        ser = pd.read_csv(in_series_name, names=colnames, header=None)

        series ={"dist": ser["dist"].values,
                  "area": ser["area"].values,
                  "out_of_scan": ser["out_of_scan"].values,
                  "percent": percents[i],
                  "scan_name": scan}

        all_series.append(series)

    # print(f"Found {len(all_series)} series for patient {patient_id}")
    if len(all_series) < 5:
        print(f"Less than five series found for patient {patient_id} exiting.")
        write_message_to_log_file(base_dir, f"Less than five series found for patient {patient_id} exiting.",
                                  patient_id, "error")
        return



    # Read json file for reference slice
    first_pid = all_series[0]["scan_name"]
    sinus_of_valsalva_info = f"{base_dir}{first_pid}/centerlines/sinus_of_valsalva_segment_max_slice_info.json"
    if not os.path.exists(sinus_of_valsalva_info):
        print(f"File {sinus_of_valsalva_info} does not exist, skipping.")
        write_message_to_log_file(base_dir, f"File {sinus_of_valsalva_info} does not exist, skipping.", patient_id, "error")
        return
    with open(sinus_of_valsalva_info, 'r') as f:
        sinus_of_valsalva_data = json.load(f)

    sinus_of_valsalva_idx = sinus_of_valsalva_data["straight_idx"]
    sinus_of_valsalva_dist_from_origin = sinus_of_valsalva_data["cl_dist"]

    if sinus_of_valsalva_idx < 40:
        print(f"Sinus of valsalva index {sinus_of_valsalva_idx} is low {sinus_of_valsalva_idx}. Warning!")
        write_message_to_log_file(base_dir, f"Sinus of valsalva index {sinus_of_valsalva_idx} is low {sinus_of_valsalva_idx}. Warning!",
                                  patient_id, "warnning")
        fig_msg += f"Sinus of valsalva index {sinus_of_valsalva_idx} is low\n"

    if sinus_of_valsalva_dist_from_origin < 20:
        print(f"Sinus of valsalva distance from origin {sinus_of_valsalva_dist_from_origin:.2f} mm is low. Warning!")
        write_message_to_log_file(base_dir, f"Sinus of valsalva distance from origin {sinus_of_valsalva_dist_from_origin:.2f} mm is low. Warning!",
                                  patient_id, "warnning")

        fig_msg += f"Sinus of valsalva distance from origin {sinus_of_valsalva_dist_from_origin:.2f} mm is low\n"

    if debug:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(12, 10)

        color = 'tab:red'
        ax1.set_xlabel('Distance (slices)')
        ax1.set_ylabel('Area (cm²)', color=color)
        for series in all_series:
            dist = series["dist"]
            area = series["area"]
            perc = series["percent"]

            ax1.plot(dist, area, label=f"CFA {perc:.2f}%", alpha=0.5)

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc="upper left")

        title = (f"Area Profiles for {patient_id}\n"
                 f"Sinus of valsalva dist from origin: {sinus_of_valsalva_dist_from_origin:.2f} mm\n")

        plt.title(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.savefig(f"{out_dir}{patient_id}_Distensability.png")
        plt.show()

    aligned_series = align_all_profiles_using_cross_correlation(base_dir, patient_id, all_series, sinus_of_valsalva_idx)
    if aligned_series is None:
        print(f"Alignment failed for patient {patient_id} exiting.")
        write_message_to_log_file(base_dir, f"Alignment failed for patient {patient_id} exiting.", patient_id, "error")
        return
    if len(aligned_series) < 5:
        print(f"Less than five final scans found for patient {patient_id} exiting.")
        write_message_to_log_file(base_dir, f"Less than five final scans found for patient {patient_id} exiting.",
                                  patient_id, "error")
        return


    if debug:
        # Plot aligned series
        fig, ax2 = plt.subplots()
        fig.set_size_inches(12, 10)
        color = 'tab:blue'
        ax2.set_xlabel('Distance (slices)')
        ax2.set_ylabel('Area (cm²)', color=color)
        for series in aligned_series:
            # dist = series["dist"]
            area = series["area"]
            perc = series["percent"]

            ax2.plot(area, label=f"CFA {perc:.2f}%", alpha=0.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc="upper right")
        title = f"Aligned Area Profiles for {patient_id}"
        plt.title(title)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.savefig(f"{out_dir}{patient_id}_Aligned_Area_Profiles.png")
        plt.show()

    measured_areas_10 = []
    measured_areas_25 = []
    measured_areas_40 = []
    measured_areas_20 = []
    measured_areas_20_mean = []
    perc = []
    for series in aligned_series:
        perc.append(series["percent"])
        area = series["area"]
        estimated_a = area[10]
        measured_areas_10.append(estimated_a)
        estimated_a = area[25]
        measured_areas_25.append(estimated_a)
        estimated_a = area[40]
        measured_areas_40.append(estimated_a)
        estimated_a = np.median(area[15:25])
        measured_areas_20.append(estimated_a)
        estimated_a = np.average(area[15:25])
        measured_areas_20_mean.append(estimated_a)


    fig_msg, measured_areas_20 = identify_series_outliers(base_dir, measured_areas_20, patient_id, fig_msg)
    dist_10 = compute_distensibility_from_series(measured_areas_10)
    dist_25 = compute_distensibility_from_series(measured_areas_25)
    dist_40 = compute_distensibility_from_series(measured_areas_40)
    dist_20 = compute_distensibility_from_series(measured_areas_20)
    dist_20_mean = compute_distensibility_from_series(measured_areas_20_mean)

    fig_msg += f"Final scans {len(aligned_series)}\n"


    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 10)
    color = 'tab:green'
    # plot the measured areas as function of the percentage
    # ax1.plot(perc, measured_areas_10, marker='o', label=f"Dist=10 Distens {dist_10:.2f}")
    # ax1.plot(perc, measured_areas_25, marker='o', label=f"Dist=25 Distens {dist_25:.2f}")
    # ax1.plot(perc, measured_areas_40, marker='x', label=f"Dist=40 Distens {dist_40:.2f}")
    ax1.plot(perc, measured_areas_20, marker='.', label=f"Dist=20 (median, outlier removed) Distens {dist_20:.2f}")
    ax1.plot(perc, measured_areas_20_mean, marker='.', label=f"Dist=20 (mean) Distens {dist_20_mean:.2f}")

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="lower right")

    ax1.set_xlabel("CFA Percentage")
    ax1.set_ylabel("Estimated Areas (cm²)")
    plt.title(f"Estimated Areas for {patient_id.rstrip('_')}\n"
              f"Sinus of valsalva dist from origin: {sinus_of_valsalva_dist_from_origin:.2f} mm idx {sinus_of_valsalva_idx}\n"
              f"{fig_msg}")
    # plt.show()
    plt.savefig(f"{out_dir}{patient_id}_Distensability.png")

    if result_f is not None:
        fig_msg = fig_msg.replace('\n',' ').replace(',',' ')
        result_f.write(f"{patient_id.rstrip('_')},{np.min(measured_areas_20_mean):.2f},{np.max(measured_areas_20_mean):.2f},"
                       f"{np.min(measured_areas_20):.2f},{np.max(measured_areas_20):.2f},"
                       f"{len(aligned_series)},{fig_msg}\n")
        result_f.flush()




def analyze_all_nstemi():
    results_out = "/storage/Data/DTU-NSTEMI/DTUImageMeasures/NSTEMI/nstemi_distensibility_results.csv"
    result_f = open(results_out, 'w')
    result_f.write("pseudo_id,min_a_20_mean,max_a_20_mean,min_a_20_median,max_a_20_median,n_scans,message\n")

    max_id = 540
    for i in range(0, max_id + 1):
        patient_id = f"NSTEMI_{i:04d}_"
        print(f"Analyzing patient {patient_id}")
        # analyze_nstemi_cfa_aorta_patient(patient_id)
        analyze_nstemi_cfa_aorta_area_profiles(patient_id, result_f)


def combine_measurements():
    """
    Combine two csv files based on the column called pseudo_id
    """
    results_in = "/storage/Data/DTUImageMeasures/NSTEMI/nstemi_distensibility_results.csv"
    clinical_in = "/storage/Data/DTU-NSTEMI/DTU-NSTEMI/Metadata/COPD_pseudoid_outcome_copy.csv"
    merged_out = "/storage/Data/DTUImageMeasures/NSTEMI/nstemi_distensibility_results_clinical.csv"

    # Read the files into two dataframes.
    df1 = pd.read_csv(results_in)
    df2 = pd.read_csv(clinical_in )

    # Merge the two dataframes, using _ID column as key
    df3 = pd.merge(df1, df2, on='pseudo_id')
    df3.set_index('pseudo_id', inplace=True)
    df3["distensability_20_median"] = (df3["max_a_20_median"] - df3["min_a_20_median"]) / (df3["min_a_20_median"] * (df3["SysBT_Scan"] - df3["DiaBT_Scan"])) * 1000

    # Write it to a new CSV file
    df3.to_csv(merged_out)

def analyze_combined_measurements():
    merged_in = "/storage/Data/DTUImageMeasures/NSTEMI/nstemi_distensibility_results_clinical.csv"

    df = pd.read_csv(merged_in)

    min_a = df["min_a_20_median"]
    max_a = df["max_a_20_median"]
    syst_bp = df["SysBT_Scan"]
    dias_bp = df["DiaBT_Scan"]
    bp_dif = syst_bp - dias_bp

    distensability = (max_a - min_a) / (min_a * bp_dif) * 1000

    plt.plot(dias_bp)
    plt.title("Diastolic BP")
    plt.show()

    plt.plot(syst_bp)
    plt.title("Systolic BP")
    plt.show()


    plt.plot(distensability)
    plt.show()

    pid = df["pseudo_id"]
    high_dist = pid[distensability > 8]
    print(high_dist)
    print(min_a[distensability > 8])
    print(max_a[distensability > 8])
    print(syst_bp[distensability > 8])
    print(dias_bp[distensability > 8])


if __name__ == '__main__':
    # analyze_cgps_2_aorta_data_max_diam_diff()
    # analyze_cgps_2_aorta_tortuosity()
    # analyze_dataset_distributions()
    # analyze_cgps_2_cfa_aorta()
    analyze_all_nstemi()
    combine_measurements()
    analyze_combined_measurements()



    # analyze_nstemi_cfa_aorta_area_profiles("CGPS-2_0049_")

    # analyze_nstemi_cfa_aorta_area_profiles("NSTEMI_0046_")
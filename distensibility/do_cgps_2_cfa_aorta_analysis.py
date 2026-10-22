import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


        step = 0.5  # mm per index
        n = min(len(s["area"]) for s in aligned_series)
        x = np.arange(n) * step
        lo, hi = 15, 25
        win_min = max(sinus_of_valsalva_idx - 50, 10)                       # alignment window
        win_max = min(sinus_of_valsalva_idx + 10, len(all_series[0]["area"]) - 10)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        for s in aligned_series:
            ax1.plot(x, s["area"][:n], alpha=0.5)
            ax2.plot(x[5:40], s["area"][5:40], ".-", alpha=0.5)
        ax1.axvspan(win_min * step, win_max * step, color="green", alpha=0.1, label="alignment window")
        ax1.axvline(sinus_of_valsalva_idx * step, color="k", ls="--", label="sinus idx (first scan)")
        for ax in (ax1, ax2):
            ax.axvspan(lo * step, hi * step, color="red", alpha=0.2, label="sampled area[15:25]")
        for i in (10, 25, 40):
            ax1.axvline(i * step, color="gray", ls=":")
        ax1.set_title("Aligned profiles"); ax2.set_title("Zoom on indices 5-40")
        ax1.set_xlabel("mm along centerline"); ax2.set_xlabel("mm along centerline")
        ax1.set_ylabel("area (mm²)"); ax1.legend()
        plt.tight_layout()
        plt.show()

        # Compute the slope and residuals for each series in the aligned series
        idx = np.arange(lo, hi)
        single_minus_median = []
        for s in aligned_series:
            a = np.asarray(s["area"][lo:hi], dtype=float)
            slope, icpt = np.polyfit(idx * step, a, 1)
            resid = a - (slope * idx * step + icpt)
            single_minus_median.append(s["area"][20] - np.median(a))
            print(f'{s["percent"]}: median {np.median(a):7.1f}  mean {a.mean():7.1f}  area[20] {s["area"][20]:7.1f}  '
                f'slope {slope:6.1f} mm²/mm  residual std {resid.std():5.1f} mm²')
        print("spread of (area[20] - window median) over phases:", np.std(single_minus_median))

        # Compare the median of the raw series with the median of the aligned series
        med_raw = np.array([np.median(s["area"][15:25]) for s in all_series])
        med_al  = np.array([np.median(s["area"][15:25]) for s in aligned_series])
        print("mean jump between neighbouring phases:", np.abs(np.diff(med_raw)).mean(), np.abs(np.diff(med_al)).mean())
        print("range:", med_raw.max() - med_raw.min(), med_al.max() - med_al.min())

        

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
    plt.close(fig)

    if result_f is not None:
        fig_msg = fig_msg.replace('\n',' ').replace(',',' ')
        result_f.write(f"{patient_id.rstrip('_')},{np.min(measured_areas_20_mean):.2f},{np.max(measured_areas_20_mean):.2f},"
                       f"{np.min(measured_areas_20):.2f},{np.max(measured_areas_20):.2f},"
                       f"{len(aligned_series)},{fig_msg}\n")
        result_f.flush()




def analyze_all_nstemi():
    results_out = "/storage/Data/DTU-NSTEMI/DTUImageMeasures/NSTEMI/nstemi_distensibility_results.csv"
    Path(os.path.dirname(results_out)).mkdir(parents=True, exist_ok=True)
    with open(results_out, 'w') as result_f:
        result_f.write("pseudo_id,min_a_20_mean,max_a_20_mean,min_a_20_median,max_a_20_median,n_scans,message\n")

        max_id = 540
        for i in range(0, max_id + 1):
            patient_id = f"NSTEMI_{i:04d}_"
            print(f"Analyzing patient {patient_id}")
            analyze_nstemi_cfa_aorta_area_profiles(patient_id, result_f)


if __name__ == '__main__':
    analyze_all_nstemi()

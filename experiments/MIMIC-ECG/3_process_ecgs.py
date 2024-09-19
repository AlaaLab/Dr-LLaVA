import os
import wfdb
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt  # Now import pyplot as plt
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm  # Import tqdm for progress bar
import logging  # Import logging for detailed logs
import argparse
import sys
import json

def find_all_rec_paths(input_dir,study_id_set=None):
    """
    Recursively find all ECG record paths by locating .hea files.

    Args:
        input_dir (str): Path to the directory containing ECG files.

    Returns:
        list: List of record paths without file extensions.
    """
    rec_paths = []
    if study_id_set==None:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.hea'):
                    rec_path = os.path.join(root, file[:-4])  # Remove '.hea' extension
                    rec_paths.append(rec_path)
    else:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.hea'):
                    if int(file[:-4]) in study_id_set:
                        rec_path = os.path.join(root, file[:-4])  # Remove '.hea' extension
                        rec_paths.append(rec_path)
    return rec_paths

        
def main():
    """
    Main function to process all ECG records using multiprocessing with a progress bar.
    """

    # Configure logging
    logging.basicConfig(
        filename='ecg_processing.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Process ECG files and save median beats plots as .jpeg images.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files',
        help="Path to the directory containing ECG files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/image_folder/',
        help="Path to the directory where output images will be saved."
    )
    parser.add_argument(
        "--cpu_percent",
        type=float,
        default=90.0,
        help="Percentage of CPU cores to use for processing (e.g., 50 for 50%)."
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default='jpeg',
        choices=['jpeg', 'png', 'pdf'],
        help="Desired image format for the output plots."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    cpu_percent = args.cpu_percent
    image_format = args.image_format

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all record paths
    with open('data/modelling_ids.json') as f: 
        study_id_set= json.load(f)

    logging.info("Scanning for ECG records...")
    rec_paths = find_all_rec_paths(input_dir, study_id_set)

     # This step helps in skipping ECGs that have already been processed
    filtered_rec_paths = []
    skipped_existing = 0

    for rec_path in tqdm(rec_paths, desc='Filtering ECG records'):
        record_name = os.path.basename(rec_path) 
        output_image_path = os.path.join(output_dir, f"{record_name}.{image_format}")
        if os.path.exists(output_image_path):
            skipped_existing += 1
            logging.info(f"Skipping {record_name}: Output image already exists.")
            continue  # Skip ECGs that have already been processed

        filtered_rec_paths.append(rec_path)

    logging.info(f"Found {len(filtered_rec_paths)} ECG records to process.")

    if not filtered_rec_paths:
        logging.warning("No ECG records found. Exiting.")
        print("No ECG records found. Exiting.")
        sys.exit(0)

    # Determine the number of CPU cores and set the number of processes based on cpu_percent
    total_cpus = cpu_count()
    num_processes = max(1, int((cpu_percent / 100.0) * total_cpus))
    logging.info(f"Using {num_processes} out of {total_cpus} CPU cores for processing ({cpu_percent}%).")
    print(f"Using {num_processes} out of {total_cpus} CPU cores for processing ({cpu_percent}%).")

    # Partial function to fix the output_dir and image_format arguments
    process_func = partial(process_record_wrapper, output_dir=output_dir, image_format=image_format)

    # Use multiprocessing Pool to process records in parallel with a progress bar
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance and to allow tqdm to update as tasks complete
        for _ in tqdm(pool.imap_unordered(process_func, filtered_rec_paths), total=len(filtered_rec_paths), desc="Processing ECG files"):
            pass

    logging.info("Processing completed.")
    print("Processing completed.")
    
def process_record_wrapper(rec_path, output_dir, image_format):
    """
    Wrapper function to pass additional arguments to process_record.

    Args:
        rec_path (str): Path to the ECG record (without file extension).
        output_dir (str): Directory where the output images will be saved.
        image_format (str): Desired image format.

    Returns:
        bool: True if processing is successful, False otherwise.
    """
    try:
        # Modify the process_record function to accept image_format
        # and save images accordingly
        # Step 1 to Step 5 are already handled in process_record
        # We'll adjust only the saving part here
        rd_record = wfdb.rdrecord(rec_path)
        
        # Get channel information
        channel_names = rd_record.sig_name
        num_leads = len(channel_names)
        
        if num_leads != 12:
            logging.warning(f"Expected 12 leads, but found {num_leads} in {rec_path}. Proceeding with available leads.")
        
        signals = rd_record.p_signal
        fs = rd_record.fs
        n_samples = signals.shape[0]
        
        try:
            annotations = wfdb.rdann(rec_path, 'atr')
            r_peaks = annotations.sample
        except FileNotFoundError:
            try:
                lead_idx = channel_names.index('II')
            except ValueError:
                lead_idx = 0
                logging.warning(f"Lead II not found in {rec_path}. Using first lead for R-peak detection.")
            
            r_peak_signal = signals[:, lead_idx]
            distance = int(0.6 * fs)
            height = np.mean(r_peak_signal) + 0.5 * np.std(r_peak_signal)
            peaks, _ = find_peaks(r_peak_signal, distance=distance, height=height)
            r_peaks = peaks
        
        pre_window = int(0.4 * fs)
        post_window = int(0.6 * fs)
        beat_length = pre_window + post_window
        
        beats_per_lead = {lead: [] for lead in channel_names}
        
        for peak in r_peaks:
            start = peak - pre_window
            end = peak + post_window
            if start >= 0 and end <= n_samples:
                beat_segment = signals[start:end, :]
                for idx, lead in enumerate(channel_names):
                    beats_per_lead[lead].append(beat_segment[:, idx])
        
        for lead in channel_names:
            beats_per_lead[lead] = np.array(beats_per_lead[lead])
        
        median_beats = {}
        for lead in channel_names:
            if beats_per_lead[lead].size == 0:
                continue
            median_beats[lead] = np.median(beats_per_lead[lead], axis=0)
        
        if not median_beats:
            logging.warning(f"No median beats computed for {rec_path}. Skipping plot.")
            return False
        
        time_axis = np.linspace(-pre_window / fs, post_window / fs, beat_length)
        
        n_rows = 3
        n_cols = 4
        
        figsize_inches = (5.12, 5.12)
        dpi = 100
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_inches, dpi=dpi)
        
        for idx, lead in enumerate(channel_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            if lead not in median_beats:
                ax.set_title(f'Lead {lead} (No Data)', fontsize=8)
                ax.axis('off')
                continue
            
            median_beat = median_beats[lead]
            ax.plot(time_axis, median_beat, color='blue', linewidth=1)
            ax.set_title(f'Lead {lead}', fontsize=10)
            ax.grid(True, linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=6)
        
        total_plots = n_rows * n_cols
        num_leads = len(channel_names)
        if num_leads < total_plots:
            for idx in range(num_leads, total_plots):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 1])
        
        record_name = os.path.basename(rec_path)
        output_image_path = os.path.join(output_dir, f"{record_name}.{image_format}")
        plt.savefig(output_image_path, format=image_format)
        plt.close(fig)
        return True
    
    except Exception as e:
        logging.error(f"Error processing {rec_path}: {e}")
        print(f"Error processing {rec_path}: {e}")
        return False

if __name__ == "__main__":
    main()




# def process_record(rec_path, output_dir):
#     """
#     Process a single ECG record: read, detect R-peaks, extract beats, compute median beats,
#     plot, and save the plot as a .jpeg image.

#     Args:
#         rec_path (str): Path to the ECG record (without file extension).
#         output_dir (str): Directory where the output images will be saved.

#     Returns:
#         bool: True if processing is successful, False otherwise.
#     """
#     try:
#         # Step 1: Read the ECG Record
#         rd_record = wfdb.rdrecord(rec_path)
        
#         # Get channel information
#         channel_names = rd_record.sig_name  # List of channel names, e.g., ['I', 'II', ..., 'V6']
#         num_leads = len(channel_names)
#         # logging.info(f"Number of leads found: {num_leads} in {rec_path}")
#         # logging.info(f"Channel names: {channel_names}")
        
#         # Verify that there are 12 leads
#         if num_leads != 12:
#             logging.warning(f"Expected 12 leads, but found {num_leads} in {rec_path}. Proceeding with available leads.")
        
#         # Extract all signals
#         signals = rd_record.p_signal  # Shape: (n_samples, n_channels)
#         fs = rd_record.fs  # Sampling frequency
#         n_samples = signals.shape[0]
        
#         # Step 2: Detect R-Peaks (Heartbeats)
#         # It's common to use a specific lead (e.g., lead II) for R-peak detection
#         try:
#             annotations = wfdb.rdann(rec_path, 'atr')
#             r_peaks = annotations.sample
#             # logging.info(f"Detected {len(r_peaks)} R-peaks from annotations in {rec_path}.")
#         except FileNotFoundError:
#             # If annotations are not available, proceed with peak detection
#             try:
#                 lead_idx = channel_names.index('II')
#             except ValueError:
#                 # If lead II is not present, default to the first lead
#                 lead_idx = 0
#                 logging.warning(f"Lead II not found in {rec_path}. Using first lead for R-peak detection.")
            
#             # Select the signal for R-peak detection
#             r_peak_signal = signals[:, lead_idx]
            
#             # Detect R-peaks using a simple peak detection algorithm
#             distance = int(0.6 * fs)  # Minimum distance between peaks (e.g., 0.6 seconds)
#             height = np.mean(r_peak_signal) + 0.5 * np.std(r_peak_signal)
#             peaks, _ = find_peaks(r_peak_signal, distance=distance, height=height)
#             r_peaks = peaks
#             # logging.info(f"Detected {len(r_peaks)} R-peaks using peak detection in {rec_path}.")
        
#         # Step 3: Segment the ECG into Individual Beats for All Leads
#         # Define a window around each R-peak (e.g., 0.4 seconds before and 0.6 seconds after)
#         pre_window = int(0.4 * fs)
#         post_window = int(0.6 * fs)
#         beat_length = pre_window + post_window
        
#         beats_per_lead = {lead: [] for lead in channel_names}
        
#         for peak in r_peaks:
#             start = peak - pre_window
#             end = peak + post_window
#             # Ensure the window is within the signal bounds
#             if start >= 0 and end <= n_samples:
#                 beat_segment = signals[start:end, :]  # Shape: (beat_length, n_channels)
#                 for idx, lead in enumerate(channel_names):
#                     beats_per_lead[lead].append(beat_segment[:, idx])
        
#         # logging.info("Extracted beats for each lead.")
        
#         # Convert lists to NumPy arrays
#         for lead in channel_names:
#             beats_per_lead[lead] = np.array(beats_per_lead[lead])
#             # logging.info(f"Lead {lead}: {beats_per_lead[lead].shape[0]} beats extracted.")
        
#         # Step 4: Compute the Median Beat for Each Lead
#         median_beats = {}
#         for lead in channel_names:
#             if beats_per_lead[lead].size == 0:
#                 # logging.warning(f"No beats extracted for lead {lead} in {rec_path}. Skipping median computation.")
#                 continue
#             median_beats[lead] = np.median(beats_per_lead[lead], axis=0)
        
#         if not median_beats:
#             logging.warning(f"No median beats computed for {rec_path}. Skipping plot.")
#             return False
        
#         # Create a time axis for plotting
#         time_axis = np.linspace(-pre_window / fs, post_window / fs, beat_length)
        
#         # Step 5: Plot the Median Beats for All 12 Leads
#         # Define the grid size for subplots (e.g., 3 rows x 4 columns)
#         n_rows = 3
#         n_cols = 4
        
#         # Define figure size and DPI to achieve 512x512 pixels
#         # Pixels = Inches * DPI => Inches = Pixels / DPI
#         desired_pixels = 512
#         dpi = 100
#         figsize_inches = (5.12, 5.12)  # 512 / 100 DPI = 5.12 inches
        
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_inches, dpi=dpi)
#         # fig.suptitle('Median Beats for 12 ECG Leads', fontsize=16)  # Adjusted title font size for smaller image
        
#         for idx, lead in enumerate(channel_names):
#             row = idx // n_cols
#             col = idx % n_cols
#             ax = axes[row, col]
            
#             if lead not in median_beats:
#                 ax.set_title(f'Lead {lead} (No Data)', fontsize=8)
#                 ax.axis('off')
#                 continue
            
#             median_beat = median_beats[lead]
#             ax.plot(time_axis, median_beat, color='blue', linewidth=1)
#             ax.set_title(f'Lead {lead}', fontsize=10)
#             # ax.set_xlabel('Time (s)', fontsize=8)
#             # ax.set_ylabel('Amplitude', fontsize=8)
#             ax.grid(True, linewidth=0.5)
#             # ax.axvline(0, color='red', linestyle='--', linewidth=0.5, label='R-peak')
#             # ax.legend(fontsize=6)
            
#             # Adjust tick parameters for smaller plots
#             ax.tick_params(axis='both', which='major', labelsize=6)
        
#         # Hide any unused subplots if less than grid size
#         total_plots = n_rows * n_cols
#         num_leads = len(channel_names)
#         if num_leads < total_plots:
#             for idx in range(num_leads, total_plots):
#                 row = idx // n_cols
#                 col = idx % n_cols
#                 axes[row, col].axis('off')
        
#         plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the main title
        
#         # Step 6: Save the Plot as a .jpeg Image
#         record_name = os.path.basename(rec_path)
#         output_image_path = os.path.join(output_dir, f"{record_name}.jpeg")
#         plt.savefig(output_image_path, format='jpeg')
#         plt.close(fig)  # Close the figure to free memory
#         # logging.info(f"Saved median beats plot as '{output_image_path}'.")
#         return True  # Indicate successful processing

# import os
# import wfdb
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from multiprocessing import Pool, cpu_count
# from functools import partial

# def find_all_rec_paths(input_dir):
#     """
#     Recursively find all ECG record paths by locating .hea files.

#     Args:
#         input_dir (str): Path to the directory containing ECG files.

#     Returns:
#         list: List of record paths without file extensions.
#     """
#     rec_paths = []
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith('.hea'):
#                 rec_path = os.path.join(root, file[:-4])  # Remove '.hea' extension
#                 rec_paths.append(rec_path)
#     return rec_paths

# def process_record(rec_path, output_dir):
#     """
#     Process a single ECG record: read, detect R-peaks, extract beats, compute median beats,
#     plot, and save the plot as a .jpeg image.

#     Args:
#         rec_path (str): Path to the ECG record (without file extension).
#         output_dir (str): Directory where the output images will be saved.
#     """
#     try:
#         # Step 1: Read the ECG Record
#         rd_record = wfdb.rdrecord(rec_path)
        
#         # Get channel information
#         channel_names = rd_record.sig_name  # List of channel names, e.g., ['I', 'II', ..., 'V6']
#         num_leads = len(channel_names)
#         # print(f"Number of leads found: {num_leads}")
#         # print(f"Channel names: {channel_names}")
        
#         # Verify that there are 12 leads
#         if num_leads != 12:
#             print(f"Warning: Expected 12 leads, but found {num_leads} in {rec_path}. Proceeding with available leads.")
        
#         # Extract all signals
#         signals = rd_record.p_signal  # Shape: (n_samples, n_channels)
#         fs = rd_record.fs  # Sampling frequency
#         n_samples = signals.shape[0]
        
#         # Step 2: Detect R-Peaks (Heartbeats)
#         # It's common to use a specific lead (e.g., lead II) for R-peak detection
#         try:
#             annotations = wfdb.rdann(rec_path, 'atr')
#             r_peaks = annotations.sample
#             # print(f"Detected {len(r_peaks)} R-peaks from annotations.")
#         except FileNotFoundError:
#             # If annotations are not available, proceed with peak detection
#             try:
#                 lead_idx = channel_names.index('II')
#             except ValueError:
#                 # If lead II is not present, default to the first lead
#                 lead_idx = 0
#                 print(f"Lead II not found in {rec_path}. Using first lead for R-peak detection.")
            
#             # Select the signal for R-peak detection
#             r_peak_signal = signals[:, lead_idx]
            
#             # Detect R-peaks using a simple peak detection algorithm
#             distance = int(0.6 * fs)  # Minimum distance between peaks (e.g., 0.6 seconds)
#             height = np.mean(r_peak_signal) + 0.5 * np.std(r_peak_signal)
#             peaks, _ = find_peaks(r_peak_signal, distance=distance, height=height)
#             r_peaks = peaks
#             # print(f"Detected {len(r_peaks)} R-peaks using peak detection.")
        
#         # Step 3: Segment the ECG into Individual Beats for All Leads
#         # Define a window around each R-peak (e.g., 0.4 seconds before and 0.6 seconds after)
#         pre_window = int(0.4 * fs)
#         post_window = int(0.6 * fs)
#         beat_length = pre_window + post_window
        
#         beats_per_lead = {lead: [] for lead in channel_names}
        
#         for peak in r_peaks:
#             start = peak - pre_window
#             end = peak + post_window
#             # Ensure the window is within the signal bounds
#             if start >= 0 and end <= n_samples:
#                 beat_segment = signals[start:end, :]  # Shape: (beat_length, n_channels)
#                 for idx, lead in enumerate(channel_names):
#                     beats_per_lead[lead].append(beat_segment[:, idx])
        
#         # print("Extracted beats for each lead.")
        
#         # Convert lists to NumPy arrays
#         for lead in channel_names:
#             beats_per_lead[lead] = np.array(beats_per_lead[lead])
#             # print(f"Lead {lead}: {beats_per_lead[lead].shape[0]} beats extracted.")
        
#         # Step 4: Compute the Median Beat for Each Lead
#         median_beats = {}
#         for lead in channel_names:
#             if beats_per_lead[lead].size == 0:
#                 # print(f"No beats extracted for lead {lead}. Skipping median computation.")
#                 continue
#             median_beats[lead] = np.median(beats_per_lead[lead], axis=0)
        
#         if not median_beats:
#             print(f"No median beats computed for {rec_path}. Skipping plot.")
#             return
        
#         # Create a time axis for plotting
#         time_axis = np.linspace(-pre_window / fs, post_window / fs, beat_length)
        
#         # Step 5: Plot the Median Beats for All 12 Leads
#         # Define the grid size for subplots (e.g., 3 rows x 4 columns)
#         n_rows = 3
#         n_cols = 4
        
#         # Define figure size and DPI to achieve 512x512 pixels
#         # Pixels = Inches * DPI => Inches = Pixels / DPI
#         desired_pixels = 512
#         dpi = 100
#         figsize_inches = (5.12, 5.12)  # 512 / 100 DPI = 5.12 inches
        
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_inches, dpi=dpi)
#         # fig.suptitle('Median Beats for 12 ECG Leads', fontsize=16)  # Adjusted title font size for smaller image
        
#         for idx, lead in enumerate(channel_names):
#             row = idx // n_cols
#             col = idx % n_cols
#             ax = axes[row, col]
            
#             if lead not in median_beats:
#                 ax.set_title(f'Lead {lead} (No Data)', fontsize=8)
#                 ax.axis('off')
#                 continue
            
#             median_beat = median_beats[lead]
#             ax.plot(time_axis, median_beat, color='blue', linewidth=1)
#             ax.set_title(f'Lead {lead}', fontsize=10)
#             # ax.set_xlabel('Time (s)', fontsize=8)
#             # ax.set_ylabel('Amplitude', fontsize=8)
#             ax.grid(True, linewidth=0.5)
#             # ax.axvline(0, color='red', linestyle='--', linewidth=0.5, label='R-peak')
#             # ax.legend(fontsize=6)
            
#             # Adjust tick parameters for smaller plots
#             ax.tick_params(axis='both', which='major', labelsize=6)
        
#         # Hide any unused subplots if less than grid size
#         total_plots = n_rows * n_cols
#         num_leads = len(channel_names)
#         if num_leads < total_plots:
#             for idx in range(num_leads, total_plots):
#                 row = idx // n_cols
#                 col = idx % n_cols
#                 axes[row, col].axis('off')
        
#         plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the main title
        
#         # Step 6: Save the Plot as a .jpeg Image
#         record_name = os.path.basename(rec_path)
#         output_image_path = os.path.join(output_dir, f"{record_name}.jpeg")
#         plt.savefig(output_image_path, format='jpeg')
#         plt.close(fig)  # Close the figure to free memory
#         # print(f"Saved median beats plot as '{output_image_path}'.")
        
#     except Exception as e:
#         print(f"Error processing {rec_path}: {e}")

# def main():
#     """
#     Main function to process all ECG records using multiprocessing.
#     """
#     import argparse
#     import sys

#     # Parse command-line arguments for flexibility
#     parser = argparse.ArgumentParser(description="Process ECG files and save median beats plots as .jpeg images.")
#     parser.add_argument(
#         "--input_dir",
#         type=str,
#         default='data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files',
#         help="Path to the directory containing ECG files."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default='data/image_folder/',
#         help="Path to the directory where output images will be saved."
#     )
#     args = parser.parse_args()

#     input_dir = args.input_dir
#     output_dir = args.output_dir

#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Find all record paths
#     print("Scanning for ECG records...")
#     rec_paths = find_all_rec_paths(input_dir)
#     print(f"Found {len(rec_paths)} ECG records to process.")

#     if not rec_paths:
#         print("No ECG records found. Exiting.")
#         sys.exit(0)

#     # Determine the number of CPU cores and set the number of processes to 50%
#     total_cpus = cpu_count()
#     num_processes = max(1, total_cpus // 2)
#     print(f"Using {num_processes} out of {total_cpus} CPU cores for processing.")

#     # Partial function to fix the output_dir argument
#     process_func = partial(process_record, output_dir=output_dir)

#     # Use multiprocessing Pool to process records in parallel
#     with Pool(processes=num_processes) as pool:
#         pool.map(process_func, rec_paths)

#     print("Processing completed.")

# if __name__ == "__main__":
#     main()



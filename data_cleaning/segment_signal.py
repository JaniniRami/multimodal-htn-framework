"""
Signal segmentation module for sleep apnea detection.

This module provides functions to:
- Parse sleep stage annotations from annotation files
- Locate apnea events in annotation files
- Segment ECG signals into fixed-duration windows with overlap
- Label segments based on apnea events and sleep stages
"""
import os
import datetime
import numpy as np
from collections import Counter


def parse_sleep_stages(file_path, recording_start_date):
    """
    Parse sleep stage annotations from an annotation file (xx.txt format).
    
    The file is expected to have a header line containing "Schlafstadium" or "Zeit",
    followed by tab-separated data with columns: [sleep_stage, time, event, duration].
    
    Args:
        file_path (str): Path to the annotation file (xx.txt)
        recording_start_date (datetime.date): The date when the recording started
        
    Returns:
        list of tuples: Each tuple is (datetime.datetime, str) representing
                       (timestamp, sleep_stage). Sleep stages include:
                       'SLEEP-S0', 'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S3', 
                       'SLEEP-S4', 'SLEEP-REM', 'N/A', 'SLEEP-MT'
    
    Note:
        Handles midnight crossovers by incrementing the date when time decreases.
    """
    common_sleep_stages = ['SLEEP-S0', 'SLEEP-S1', 'SLEEP-S2', 'SLEEP-S3', 'SLEEP-S4', 'SLEEP-REM', 'N/A', 'SLEEP-MT']
    sleep_stages = []
    current_date = recording_start_date
    previous_time = None
    
    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()
        # Find header line - try "Schlafstadium" first, fallback to "Zeit"
        try:
            header_index = next(
                i for i, line in enumerate(lines) if ("Schlafstadium" in line)
            )
        except StopIteration:
            header_index = next(i for i, line in enumerate(lines) if ("Zeit" in line))

        # Parse sleep stage annotations starting after header
        for line in lines[header_index + 1 :]:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            try:
                stage = parts[0]
                assert stage in common_sleep_stages, f"Invalid sleep stage: {stage} - file: {file_path}"
                time_str = parts[1]
                time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
                
                # Handle midnight reset: if time decreases, increment date
                if previous_time and time_obj < previous_time:
                    current_date += datetime.timedelta(days=1)
                
                full_datetime = datetime.datetime.combine(current_date, time_obj)
                sleep_stages.append((full_datetime, stage))
                previous_time = time_obj
            except ValueError:
                continue  # Skip invalid lines

    return sleep_stages


def get_interpolated_sleep_stage(segment_start_time, segment_end_time, sleep_stages, expected_interval=30, missing_threshold=0.2):
    """
    Interpolates sleep stage for a segment based on sleep stage annotations.

    The function:
    1. Creates expected time slots based on the expected interval
    2. Maps known sleep stage annotations to the closest expected time slot
    3. Interpolates missing values using nearest neighbor information
    4. Returns the most frequent sleep stage in the segment

    Parameters:
        segment_start_time (datetime.datetime): Start time of the segment.
        segment_end_time (datetime.datetime): End time of the segment.
        sleep_stages (list of tuples): Each tuple is (time, stage) where time is a datetime.
        expected_interval (int): Expected interval between annotations (in seconds), default is 30.
        missing_threshold (float): Maximum fraction of missing annotations allowed (0.0-1.0);
                                   if exceeded, the segment is discarded and None is returned.

    Returns:
        str or None: The final sleep stage (e.g. 'SLEEP-S1') or None if the segment is discarded
                    due to too many missing annotations.
    """
    # Calculate total duration and expected number of annotations
    duration = (segment_end_time - segment_start_time).total_seconds()
    num_expected = int(round(duration / expected_interval)) + 1
    expected_times = [segment_start_time + datetime.timedelta(seconds=i * expected_interval) for i in range(num_expected)]
    
    # Create list for sleep stages corresponding to each expected time
    stages_list = [None] * num_expected
    tolerance = expected_interval / 2  # e.g., 12.5 seconds

    # Assign known sleep stages to the closest expected time slot if within tolerance
    for time_obj, stage in sleep_stages:
        if segment_start_time <= time_obj <= segment_end_time:
            # Find the closest expected timestamp
            diffs = [abs((time_obj - t).total_seconds()) for t in expected_times]
            idx = diffs.index(min(diffs))
            if min(diffs) <= tolerance:
                stages_list[idx] = stage

    # Check for too many missing annotations
    missing_count = sum(1 for s in stages_list if s is None)
    if missing_count > missing_threshold * num_expected:
        return None  # Discard segment

    # Interpolate missing values using neighbor information
    for i in range(num_expected):
        if stages_list[i] is None:
            left = None
            right = None
            # Look left for nearest known stage
            for j in range(i - 1, -1, -1):
                if stages_list[j] is not None:
                    left = stages_list[j]
                    break
            # Look right for nearest known stage
            for j in range(i + 1, num_expected):
                if stages_list[j] is not None:
                    right = stages_list[j]
                    break
            # If both neighbors are available and equal, use that value; 
            # otherwise, prefer the left neighbor (closer to start of segment)
            if left is not None and right is not None:
                stages_list[i] = left if left == right else left
            elif left is not None:
                stages_list[i] = left
            elif right is not None:
                stages_list[i] = right

    # Determine final sleep stage by taking the mode (most frequent stage)
    final_stage, _ = Counter(stages_list).most_common(1)[0]
    return final_stage


def locate_apnea_events(annotation_file_path, signal_start_dt, LOOKFOR):
    """
    Locate apnea events in an annotation file.
    
    Parses the annotation file to find all events matching the types in LOOKFOR.
    Events are expected to be in tab-separated format with columns:
    [sleep_stage, time, event_type, duration].
    
    Args:
        annotation_file_path (str): Path to the annotation file (xx.txt)
        signal_start_dt (datetime.datetime): Start datetime of the signal recording
        LOOKFOR (list or set): List of event types to search for (e.g., 
                              ['APNEA-OBSTRUCTIVE', 'APNEA-CENTRAL', 'HYPOPNEA'])
    
    Returns:
        tuple: (apnea_events, ann_start_time) where:
            - apnea_events: List of dictionaries, each containing:
                - 'event_type': str (e.g., 'APNEA-OBSTRUCTIVE')
                - 'start_time': datetime.datetime
                - 'end_time': datetime.datetime
                - 'duration': int (seconds)
            - ann_start_time: datetime.datetime of the first annotation in the file
    
    Note:
        Handles midnight crossovers by incrementing the date when time decreases.
    """
    apnea_events = []
    with open(annotation_file_path, "r", encoding='latin-1') as file:
        lines = file.readlines()

        fix_shift = 0
        try:
            header_index = next(
                i for i, line in enumerate(lines) if ("Schlafstadium" in line)
            )
        except StopIteration:
            header_index = next(i for i, line in enumerate(lines) if ("Zeit" in line))
            fix_shift = 1

        # Extract first annotation line to determine annotation start time
        first_ann_line = lines[header_index + 1].split("\t")
        # fix_shift accounts for different header formats (0 for "Schlafstadium", 1 for "Zeit")
        # Column indices: [sleep_stage, time, event, duration] when fix_shift=0
        #                  [time, event, duration] when fix_shift=1

        ann_start_time = datetime.datetime.strptime(
            signal_start_dt.date().strftime("%d.%m.%y")
            + " "
            + first_ann_line[1 - fix_shift],
            "%d.%m.%y %H:%M:%S",
        )

        prev_time = ann_start_time

        for line in lines[header_index + 1 :]:
            line_data = line.split("\t")
            start_time = line_data[1 - fix_shift]
            sleep_event = line_data[2 - fix_shift]
            duration = line_data[3 - fix_shift]
            current_time = datetime.datetime.strptime(
                signal_start_dt.date().strftime("%d.%m.%y") + " " + start_time,
                "%d.%m.%y %H:%M:%S",
            )
            if current_time < prev_time:
                current_time = current_time + datetime.timedelta(days=1)

            if sleep_event in LOOKFOR:
                duration = int(duration)
                end_time = current_time + datetime.timedelta(seconds=duration)
                apnea_event = {
                    "event_type": sleep_event,
                    "start_time": current_time,
                    "end_time": end_time,
                    "duration": duration,
                }
                apnea_events.append(apnea_event)

    return apnea_events, ann_start_time

def get_segments(annotation_file_path, ecg_signal, edf_header, LOOKFOR, segment_duration, signal_fs=200):
    """
    Segment ECG signal into fixed-duration windows with overlap and label them.
    
    This function:
    1. Parses sleep stages and apnea events from the annotation file
    2. Segments the ECG signal into overlapping windows
    3. Labels each segment based on whether it contains apnea events
    4. Associates each segment with its sleep stage
    
    Args:
        annotation_file_path (str): Path to annotation file (xx.txt)
        ecg_signal (np.ndarray): ECG signal array
        edf_header (dict): EDF file header containing 'start_date' and 'start_time' keys
                          Format: {'start_date': 'DD.MM.YY', 'start_time': 'HH.MM.SS'}
        LOOKFOR (list or set): Event types to search for (e.g., apnea types)
        segment_duration (int): Duration of each segment in seconds
        signal_fs (int): Sampling frequency of the signal in Hz (default: 200)
    
    Returns:
        tuple: (signal_segments_report, signal_segments, segments_labels, 
                sleep_stage_per_segment, apnea_events_found, non_apnea_events_found)
            - signal_segments_report: List of dicts with segment metadata
            - signal_segments: List of np.ndarray, each is a segment of the signal
            - segments_labels: List of int (0 or 1), 1 if segment contains apnea
            - sleep_stage_per_segment: List of str, sleep stage for each segment
            - apnea_events_found: int, count of segments with apnea
            - non_apnea_events_found: int, count of segments without apnea
    
    Note:
        - Segments have 30% overlap by default
        - Segments with sleep stages 'SLEEP-MT', 'UNKNOWN', 'N/A', or None are skipped
        - A segment is labeled as apnea (1) if any apnea event overlaps with the segment
          (event starts in segment, ends in segment, or completely contains the segment)
    """
    signal_start_dt = datetime.datetime.strptime(
        edf_header["start_date"] + " " + edf_header["start_time"], "%d.%m.%y %H.%M.%S"
    )

    extracted_sts = parse_sleep_stages(annotation_file_path ,signal_start_dt)

    apnea_events, ann_start_time = locate_apnea_events(annotation_file_path, signal_start_dt, LOOKFOR)

    # Calculate signal start index (currently unused but kept for potential future use)
    # This would skip the signal to align with annotation start time
    signal_start_index = int(
        (abs(ann_start_time - signal_start_dt)).total_seconds() * signal_fs
    )

    signal_segments_report = []
    signal_segments = []
    segments_labels = []
    sleep_stage_per_segment = []

    segment_size = segment_duration * signal_fs 
    overlap_size = int(segment_size * 0.3)  # 30% overlap between segments

    apnea_events_found = 0
    non_apnea_events_found = 0

    for j in range(0, len(ecg_signal) - segment_size + 1, segment_size - overlap_size):

        start_idx = j
        end_idx = min(j + segment_size, len(ecg_signal))

        signal_segment = ecg_signal[start_idx:end_idx]

        # Note: Zero percentage check is commented out but kept for reference
        # Could be used to filter out segments with too many zeros (artifacts)

        # Ensure segment has expected size (may be smaller for last segment)
        # For now, we assert it must be exact size, but last segment might need special handling
        assert len(signal_segment) == segment_size, f"Segment size is not correct, size:{len(signal_segment)}"

        segment_start_time = signal_start_dt + datetime.timedelta(seconds=j / signal_fs)
        segment_end_time = segment_start_time + datetime.timedelta(seconds=segment_duration)

        sleep_stage = get_interpolated_sleep_stage(segment_start_time, segment_end_time, extracted_sts)

        # Skip segments with invalid or missing sleep stages
        if sleep_stage == "SLEEP-MT":
            print(f"Sleep stage not found for segment {segment_start_time} - {segment_end_time}, SLEEP-MT")
            continue
        if sleep_stage == "UNKNOWN":
            print(f"Sleep stage not found for segment {segment_start_time} - {segment_end_time}, UNKNOWN")
            continue
        if sleep_stage == "N/A":
            print(f"Sleep stage not found for segment {segment_start_time} - {segment_end_time}, N/A")
            continue
        if sleep_stage is None:
            print(f"Sleep stage not found for segment {segment_start_time} - {segment_end_time}")
            continue
        apnea_binary_class = False
        apnea_events_in_segment = []

        # Check if any apnea event overlaps with this segment
        # An event overlaps if:
        # 1. Event starts within the segment, OR
        # 2. Event ends within the segment, OR
        # 3. Event completely contains the segment
        for event in apnea_events:
            event_start_time = event["start_time"]
            event_end_time = event["end_time"]

            # Check if event overlaps with segment
            # Event overlaps if: (starts in segment) OR (ends in segment) OR (contains segment)
            event_overlaps = (
                (segment_start_time <= event_start_time <= segment_end_time) or
                (segment_start_time <= event_end_time <= segment_end_time) or
                (event_start_time <= segment_start_time and event_end_time >= segment_end_time)
            )

            if event_overlaps:
                apnea_binary_class = True
                
                apnea_event_type = event["event_type"]
                # Clip event times to segment boundaries
                apnea_event_start_time = max(event_start_time, segment_start_time)
                apnea_event_end_time = min(event_end_time, segment_end_time)
                # Duration is the original event duration (not clipped)
                apnea_event_duration = event["duration"]

                apnea_event_segment = {
                    "sleep_stage": sleep_stage,
                    "event_type": apnea_event_type,
                    "start_time": apnea_event_start_time,
                    "end_time": apnea_event_end_time,
                    "duration": apnea_event_duration,
                }
                apnea_events_found += 1
                apnea_events_in_segment.append(apnea_event_segment)
           


        if apnea_binary_class == False:
            non_apnea_events_found += 1
           

        segment_report = {
            "segment_start_time": segment_start_time.strftime("%H:%M:%S"),
            "segment_end_time": segment_end_time.strftime("%H:%M:%S"),
            "is_apnea_event": apnea_binary_class,
            "apnea_details": apnea_events_in_segment,
        }

       
        signal_segments_report.append(segment_report)
        signal_segments.append(signal_segment)
        segments_labels.append(1 if apnea_binary_class else 0)
        sleep_stage_per_segment.append(sleep_stage)

    assert len(signal_segments) == len(segments_labels) == len(sleep_stage_per_segment), \
        'Lengths of signal segments, labels and sleep stages per segment must be equal'

    print("Apnea events found: ", apnea_events_found)
    print("Non-apnea events found: ", non_apnea_events_found)
    return (
        signal_segments_report,
        signal_segments,
        segments_labels,
        sleep_stage_per_segment,
        apnea_events_found,
        non_apnea_events_found,
    )

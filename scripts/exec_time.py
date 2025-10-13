import re
import os
import sys

def extract_timestamps_from_log(log_file_path):
    """
    Extract timestamps from 'Number of variables' and 'Problem solved in ...' lines in an error.log file.
    Args:
        log_file_path (str): Path to the error.log file.
    Returns:
        int: Difference between the largest and smallest timestamps in seconds.
    """
    # Regular expressions to match timestamps
    timestamp_pattern = r"\d{10}\.\d{6}"

    first_timestamp = None
    last_timestamp = None

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            if "Number of variables" in line:
                # Extract timestamp from 'Number of variables' line
                match = re.search(timestamp_pattern, line)
                if match:
                    first_timestamp = float(match.group())
            elif "Problem solved in" in line:
                # Extract timestamp from 'Problem solved in ...' line
                match = re.search(timestamp_pattern, line)
                if match:
                    last_timestamp = float(match.group())

    if first_timestamp and last_timestamp:
        return first_timestamp, last_timestamp
    else:
        return


def check_log_file(log_file_path):
    try:
        with open(log_file_path, "r") as file:
            line_count = sum(1 for _ in file)
            if line_count > 13 or line_count < 12:
                print(f"File error.log for {log_file_path} isn't complete.")
                return None
    except FileNotFoundError:
        print(log_file_path.split("/"))
        print(f"No error.log for {log_file_path.split('/')}")
        return False


def get_full_exec_time(directory):
    """
    Calculate the full execution time for all error.log files in subdirectories.
    Args:
        directory (str): Path to the main directory.
    Returns:
        int: Total execution time in seconds.
    """
    # Initialize variables to store timestamps
    # smallest_timestamp = float("inf")
    # largest_timestamp = float("-inf")
    total_exec_time = 0

    for root, _, _ in os.walk(directory):
        if "output" in root:
            for _, _, out_files in os.walk(root):
                for filename in out_files:
                    if filename == "error.log":
                        log_file_path = os.path.join(root, filename)

                        check_log_file(log_file_path)

                        timestamps = extract_timestamps_from_log(log_file_path)

                        if timestamps:
                            first_timestamp, last_timestamp = timestamps

                            dir_time = last_timestamp - first_timestamp
                            total_exec_time += dir_time

                            # update timestamps
                            # smallest_timestamp = min(smallest_timestamp, first_timestamp)
                            # largest_timestamp = max(largest_timestamp, last_timestamp)

    # total_exec_time = int(largest_timestamp - smallest_timestamp)
    return total_exec_time

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} {{full,script,family}} <demand_path>')
        exit()
        
    option = sys.argv[1]
    main_directory = sys.argv[2]

    if option == 'full':
        full_exec_time = get_full_exec_time(main_directory)
        if full_exec_time is not None:
            minutes_time = full_exec_time / 60
            hours_time = (full_exec_time / 60) / 60
            print(f"Total execution time: {minutes_time:.2f} minutes | {hours_time:.2f} hours | {hours_time / 24:.2f} days.")
    elif option == 'script':
        full_exec_time = get_full_exec_time(main_directory)
        if full_exec_time is not None:
            minutes_time = full_exec_time / 60
            print(f'{minutes_time:.2f}')
    elif option == 'family':
        family = sys.argv[3]
        log_file_path = os.path.join(main_directory, f'{family}/output/error.log')
        timestamps = extract_timestamps_from_log(log_file_path)

        if timestamps:
            first_timestamp, last_timestamp = timestamps
            minutes_time = (last_timestamp - first_timestamp) / 60
            hours_time = minutes_time / 60
            print(f"E.T.: {minutes_time:.2f}min | {hours_time:.2f}h | {hours_time / 24:.2f}d")

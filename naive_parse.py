import re
import argparse

def parse_log(log_lines, target_format):
    pattern = re.compile(r'step:\d+(\s+(\S+):(\S+))+')
    results = []

    for line in log_lines:
        if target_format in line:
            segments = line.split(' ')
            for seg in segments:
                if target_format in seg:
                    results.append(seg)

    return results

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse log file and extract metric values.")
    parser.add_argument("--file_path", type=str, help="Path to the log file")
    parser.add_argument("--target_format", type=str, help="Target metric format to extract")
    args = parser.parse_args()

    log_lines = read_log_file(args.file_path)
    selected_values = parse_log(log_lines, args.target_format)
    print(selected_values)

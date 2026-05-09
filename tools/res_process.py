import argparse
import re

def extract_processing_time(file_path):
    pattern = r'^>>Processing finished in (\d+\.\d+|\d+) \(ms\)\.$'
    numbers = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip() 
            match = re.match(pattern, line)
            if match:
                numbers.append(match.group(1))
    
    return numbers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract processing times from a log file.")
    parser.add_argument("file_path", help="Path to the input file")
    args = parser.parse_args()

    result = extract_processing_time(args.file_path)
    print("Extracted numbers:")
    for num in result:
        print(num)
    print("Sum: ", sum(float(x) for x in result))

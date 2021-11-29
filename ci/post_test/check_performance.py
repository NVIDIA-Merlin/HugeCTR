import os
from argparse import ArgumentParser
import json
import re
import glob
from collections import defaultdict


expected_result_json = './ci/post_test/perf_benchmark.json'
log_pattern = {
    'wdl_8gpu': {
        'cmd_log': r'python3 /workdir/test/pybind_test/single_node_test.py --json-file=/workdir/test/scripts/wdl_8gpu.json',
        'result_log': r'Finish 3000 iterations with batchsize: 16384 in (.*)s'
    },
    'dlrm_1node': {
        'cmd_log': r'python3 /workdir/samples/dlrm/dgx_a100.py',
        'result_log': r'Hit target accuracy AUC 0.802500 at 68274 / 75868 iterations with batchsize 55296 in (.*)s. Average'
    },
    'dlrm_14node': {
        'cmd_log': r'HugeCTR Version',
        'result_log': r'Hit target accuracy AUC 0.802500 at 58520 / 58527 iterations with batchsize 71680 in (.*)s. Average'
    },
    'inference_benchmark': {
        'cmd_log': r'Server:',
        'result_log': r'Avg request latency: (.*?) usec'
    }
}

def extract_result_from_log(job_name, log_path):
    log_files = glob.glob(os.path.join(log_path, "*", "results", "*.log"))
    log_files = [fname for fname in log_files if re.match(r".*[0-9]+.log", fname)]
    print("all log files", log_files)
    latest_log_file = max(log_files, key=os.path.getctime)
    print("use latest log file", latest_log_file)
    job_log_pattern = log_pattern[job_name]
    with open(latest_log_file, 'r', errors='ignore') as f:
        lines = ''.join(f.readlines())
        job_logs = lines.split('+ ')
        for each_job_log in job_logs:
            if re.search(job_log_pattern['cmd_log'], each_job_log):
                match = re.search(job_log_pattern['result_log'], each_job_log)
                print(match.group(1))
                result = eval(match.group(1))
                return result

def extract_result_from_json(job_name):
    with open(expected_result_json, 'r') as f:
        expected_reuslt = json.load(f)
    return expected_reuslt[job_name]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--job_name')
    parser.add_argument('--log_path')
    args = parser.parse_args()

    perf_result = extract_result_from_log(args.job_name, args.log_path)
    expected_result = extract_result_from_json(args.job_name)

    if float(perf_result) > float(expected_result):
        raise RuntimeError("performance get worse. perf_result:{} vs. expected result:{}".format(perf_result, expected_result))
    else:
        print("performance check pass. perf_result:{} vs. expected result:{}".format(perf_result, expected_result))

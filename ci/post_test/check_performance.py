import os
from argparse import ArgumentParser
import json
import re
import glob
from collections import defaultdict
import math

expected_result_json = './ci/post_test/perf_benchmark.json'
log_pattern = {
    'wdl_8gpu': {
        'cmd_log': r'python3 /workdir/test/pybind_test/single_node_test.py --json-file=/workdir/test/scripts/wdl_8gpu.json',
        'result_log': r'Finish 3000 iterations with batchsize: 16384 in (\d+\.?\d*)s'
    },
    'dlrm_1node': {
        'cmd_log': r'python3 /workdir/samples/dlrm/dgx_a100.py',
        'result_log': r'Hit target accuracy AUC 0.802500 at 68274 / 75868 iterations with batchsize 55296 in (\d+\.?\d*)s. Average'
    },
    'dlrm_14node': {
        'cmd_log': r'HugeCTR Version',
        'result_log': r'Hit target accuracy AUC 0.802500 at 58520 / 58527 iterations with batchsize 71680 in (\d+\.?\d*)s. Average'
    },
    'inference_benchmark': {
        'cmd_log': r'Server:',
        'result_log': r'Avg request latency: (\d+\.?\d*) usec'
    },
    'sok': {
        'cmd_log': r'python3 main.py ',
        'result_log': r'elapsed_time: (\d+\.?\d*)'
    },
    'train_bmk': {
        'cmd_log': r'python3 ./benchmark_train.py',
        'result_log': r'Time\(200 iters\): (\d+\.?\d*)s'
    },
    'inference_benchmark_avg': {
        'cmd_log': r'Client:',
        'result_log': r'Avg latency: (\d+\.?\d*) usec'
    },
    'inference_benchmark_p50': {
        'cmd_log': r'Client:',
        'result_log': r'p50 latency: (\d+\.?\d*) usec'
    },
    'inference_benchmark_p90': {
        'cmd_log': r'Client:',
        'result_log': r'p90 latency: (\d+\.?\d*) usec'
    },
    'inference_benchmark_p95': {
        'cmd_log': r'Client:',
        'result_log': r'p95 latency: (\d+\.?\d*) usec'
    },
    'inference_benchmark_p99': {
        'cmd_log': r'Client:',
        'result_log': r'p99 latency: (\d+\.?\d*) usec'
    }
}

def extract_result_from_log(job_name, log_path):
    log_files = glob.glob(os.path.join(log_path, "*", "results", "*.log"))
    log_files = [fname for fname in log_files if re.match(r".*[0-9]+.log", fname)]
    print("all log files", log_files)
    latest_log_file = max(log_files, key=os.path.getctime)
    print("use latest log file", latest_log_file)
    job_log_pattern = log_pattern[job_name]
    results = []
    with open(latest_log_file, 'r', errors='ignore') as f:
        lines = ''.join(f.readlines())
        job_logs = lines.split('+ ')
        for each_job_log in job_logs:
            if re.search(job_log_pattern['cmd_log'], each_job_log):
                for line in each_job_log.split('\n'):
                    match = re.search(job_log_pattern['result_log'], line)
                    if match is None:
                        continue
                    result = float(match.group(1))
                    results.append(result)
    return sum(results) / len(results) if len(results) > 0 else float('inf')

def extract_result_from_json(job_name):
    with open(expected_result_json, 'r') as f:
        expected_reuslt = json.load(f)
    return expected_reuslt[job_name]

def collect_benchmark_result(log_path):
    headers = ['name', 'batch_size', 'batch_size_per_gpu', 'total_gpu_num', 'node_num', 'precision', 'platform', 'ms per iteration', 'p99 latency(usec)', 'p95 latency(usec)', 'p90 latency(usec)', 'p50 latency(usec)', 'Avg latency(usec)', 'result_log_path']
    list_benchmark = []
    for train_bmk_name in ['wdl', 'dcn', 'deepfm']:
        for bz_per_gpu in [256, 512, 1024, 2048, 4096, 8192]:
            for gpu_num in [1, 2, 4, 8, 16, 32]:
                for mixed_precision in ['FP16', 'FP32']:
                    benchmark = ['' for _ in range(len(headers))]
                    benchmark[headers.index('name')] = train_bmk_name
                    benchmark[headers.index('batch_size')] = bz_per_gpu * gpu_num
                    benchmark[headers.index('batch_size_per_gpu')] = bz_per_gpu
                    benchmark[headers.index('total_gpu_num')] = gpu_num
                    node_num =  (gpu_num - 1) // 8 + 1
                    benchmark[headers.index('node_num')] = node_num
                    benchmark[headers.index('precision')] = mixed_precision
                    benchmark[headers.index('platform')] = 'selene'

                    gpu_num_per_node = gpu_num % 8 if gpu_num % 8 != 0 else 8
                    result_log_path = os.path.join(log_path, 'train_benchmark--{bmk_name}--{node_num}x{gpu_num_per_node}x{bz_per_gpu}x{mixed_precision}'.format(
                        bmk_name = train_bmk_name,
                        node_num = node_num,
                        gpu_num_per_node = gpu_num_per_node,
                        bz_per_gpu = bz_per_gpu,
                        mixed_precision = mixed_precision
                    ))
                    benchmark[headers.index('result_log_path')] = result_log_path
                    if os.path.exists(result_log_path):
                        ms_per_iteration = extract_result_from_log('train_bmk', result_log_path)
                        ms_per_iteration = ms_per_iteration / 200 * 1000
                        benchmark[headers.index('ms per iteration')] = ms_per_iteration
                    list_benchmark.append(benchmark)

    for bz in [1, 32, 64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        for mixed_precision in ['FP16', 'FP32']:
            benchmark = ['' for _ in range(len(headers))]
            benchmark[headers.index('name')] = 'inference_benchmark'
            benchmark[headers.index('batch_size')] = bz
            benchmark[headers.index('batch_size_per_gpu')] = bz
            benchmark[headers.index('total_gpu_num')] = 1
            benchmark[headers.index('node_num')] = 1
            benchmark[headers.index('precision')] = mixed_precision
            benchmark[headers.index('platform')] = 'selene'

            result_log_path = os.path.join(log_path, 'inference_benchmark_{bz}x{mixed_precision}'.format(
                bz=bz,
                mixed_precision=mixed_precision
            ))
            benchmark[headers.index('result_log_path')] = result_log_path
            if os.path.exists(result_log_path):
                avg_latency = extract_result_from_log('inference_benchmark_avg', result_log_path)
                benchmark[headers.index('Avg latency(usec)')] = avg_latency
                p50_latency = extract_result_from_log('inference_benchmark_p50', result_log_path)
                benchmark[headers.index('p50 latency(usec)')] = p50_latency
                p90_latency = extract_result_from_log('inference_benchmark_p90', result_log_path)
                benchmark[headers.index('p90 latency(usec)')] = p90_latency
                p95_latency = extract_result_from_log('inference_benchmark_p95', result_log_path)
                benchmark[headers.index('p95 latency(usec)')] = p95_latency
                p99_latency = extract_result_from_log('inference_benchmark_p99', result_log_path)
                benchmark[headers.index('p99 latency(usec)')] = p99_latency
            list_benchmark.append(benchmark)
    
    for bz in [8192, 16384, 32768, 65536]:
        for gpu_num in [1, 2, 4, 8]:
            benchmark = ['' for _ in range(len(headers))]
            benchmark[headers.index('name')] = 'sok'
            benchmark[headers.index('batch_size')] = bz
            bz_per_gpu = bz // gpu_num
            benchmark[headers.index('batch_size_per_gpu')] = bz_per_gpu
            benchmark[headers.index('total_gpu_num')] = gpu_num
            benchmark[headers.index('node_num')] = 1
            benchmark[headers.index('precision')] = 'FP32'
            benchmark[headers.index('platform')] = 'selene'

            result_log_path = os.path.join(log_path, 'sok_benchmark_{bz_per_gpu}x{gpu_num}'.format(
                bz_per_gpu=bz_per_gpu,
                gpu_num=gpu_num
            ))
            benchmark[headers.index('result_log_path')] = result_log_path
            if os.path.exists(result_log_path):
                ms_per_iteration = extract_result_from_log('sok', result_log_path)
                ms_per_iteration = ms_per_iteration * 10
                benchmark[headers.index('ms per iteration')] = ms_per_iteration
            list_benchmark.append(benchmark)
    print(','.join(headers))
    for benchmark in list_benchmark:
        print(','.join(str(i) for i in benchmark))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--collect_result', action='store_true', default=False)
    parser.add_argument('--job_name')
    parser.add_argument('--log_path', required=True)
    args = parser.parse_args()

    if args.collect_result:
        collect_benchmark_result(args.log_path)
    else:
        perf_result = extract_result_from_log(args.job_name, args.log_path)
        expected_result = extract_result_from_json(args.job_name)

        if float(perf_result) > float(expected_result):
            raise RuntimeError("performance get worse. perf_result:{} vs. expected result:{}".format(perf_result, expected_result))
        else:
            print("performance check pass. perf_result:{} vs. expected result:{}".format(perf_result, expected_result))

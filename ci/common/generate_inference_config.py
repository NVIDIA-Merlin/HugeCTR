from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("--config_template", type=str)
parser.add_argument("--ps_template", type=str)
parser.add_argument("--batchsize", type=str)
parser.add_argument("--mixed_precision", type=str)
parser.add_argument("--config_output", type=str)
parser.add_argument("--ps_output", type=str)
args = parser.parse_args()

with open(args.config_template, "r") as f:
    config_pbtxt_template = f.readlines()
    config_pbtxt_template = "".join(config_pbtxt_template)

config_pbtxt = config_pbtxt_template.replace("%%batchsize", args.batchsize).replace(
    "%%mixed_precision", args.mixed_precision
)
with open(args.config_output, "w") as f:
    f.write(config_pbtxt)

with open(args.ps_template, "r") as f:
    ps_json_template = json.load(f)


def str2bool(v):
    return v.lower() in ("true")


ps_json_template["models"][0]["max_batch_size"] = args.batchsize
ps_json_template["models"][0]["mixed_precision"] = str2bool(args.mixed_precision)

with open(args.ps_output, "w") as f:
    json.dump(ps_json_template, f)

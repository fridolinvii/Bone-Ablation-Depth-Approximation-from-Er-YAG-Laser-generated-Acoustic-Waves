import pathlib
import sys
import time

from hyperband.definitions.parameters_conv import get_params, try_params
from hyperband.hyperband import Hyperband
from parameter import run_args


def search_best_hyperparams(fixed_args):

    hb = Hyperband(get_params, try_params)
    hb.run(skip_last=1, fixed_params=fixed_args)
    print('all done')


if __name__ == "__main__":

    args = run_args.parser.parse_args()

    if args.logfile is True:
        print("write output to logfile")
        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_file_path = args.path_to_folder+'/results/hyperband/results_{}.log'.format(time_string)
        pathlib.Path(args.path_to_folder+'/results/hyperband').mkdir(parents=True, exist_ok=True)
        print("Check log file in: ")
        print(log_file_path)
        sys.stdout = open(log_file_path, 'w')
    else:
        print("write output to console only")

    search_best_hyperparams(args)

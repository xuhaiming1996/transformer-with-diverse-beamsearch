import argparse
import logging
import os
import sys
import time
import uuid
import warnings

import zmq
from termcolor import colored
from zmq.utils import jsonapi
from bert_serving.server.page.fairseq import options

__all__ = ['set_logger', 'send_ndarray', 'send_PAGE' ,'get_args_parser',
          'auto_bind', 'import_tf', 'TimeContext']


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


def send_PAGE(src, dest, res, req_id=b'', flags=0, copy=True, track=False):


    return src.send_multipart([dest, jsonapi.dumps(res), b'', req_id], flags, copy=copy, track=track)



def send_ndarray(src, dest, X, req_id=b'', flags=0, copy=True, track=False):
    '''
                send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)


    :param src: sink_embed
    :param dest: r['client_id']
    :param X:
    :param req_id:
    :param flags:
    :param copy:
    :param track:
    :return:
    '''
    """send a numpy array with metadata"""
    md = dict(dtype=str(X.dtype), shape=X.shape)
    '''
    send_multipart([client_id, jsonapi.dumps([f.tokens for f in tmp_f]),
                                                 b'', ServerCmd.data_token])
    '''
    return src.send_multipart([dest, jsonapi.dumps(md), X, req_id], flags, copy=copy, track=track)


def check_max_seq_len(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        ivalue = int(value)
        if ivalue <= 3:
            raise argparse.ArgumentTypeError("%s is an invalid int value must be >3 "
                                             "(account for maximum three special symbols in BERT model) or NONE" % value)
    except TypeError:
        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
    return ivalue


def get_args_parser():
    from bert_serving.server.page.fairseq import options

    parser = options.get_generation_parser(interactive=True)
    # parser = parser.add_argument_group('Serving Configs',
    #                                    'config how server utilizes GPU/CPU resources')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for sending result to client')
    parser.add_argument('-http_port', type=int, default=None,
                        help='server port for receiving HTTP requests')
    parser.add_argument('-http_max_connect', type=int, default=10,
                        help='maximum number of concurrent HTTP connections')
    parser.add_argument('-cors', type=str, default='*',
                        help='setting "Access-Control-Allow-Origin" for HTTP requests')
    parser.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    parser.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    parser.add_argument('-priority_batch_size', type=int, default=16,
                        help='batch smaller than this size will be labeled as high priority,'
                             'and jumps forward in the job queue')
    parser.add_argument('-cpu', action='store_true', default=False,
                        help='running on CPU (default on GPU)')
    parser.add_argument('-xla', action='store_true', default=False,
                        help='enable XLA compiler (experimental)')
    parser.add_argument('-fp16', action='store_true', default=False,
                        help='use float16 precision (experimental)')
    parser.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory \
                            that each visible GPU should be allocated per worker. \
                            Should be in range [0.0, 1.0]')
    parser.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0). \
                            If num_worker > len(device_map), then device will be reused; \
                            if num_worker < len(device_map), then device_map[:num_worker] will be used')
    parser.add_argument('-prefetch_size', type=int, default=10,
                        help='the number of batches to prefetch on each worker. When running on a CPU-only machine, \
                            this is set to 0 for comparability')

    parser.add_argument('-verbose', action='store_true', default=True,
                        help='turn on tensorflow logging for debug')

    # parser.add_argument('--bpe-codes', type=str,
    #                     help='BPE code')
    #
    # parser.add_argument('--source-lang', type=str, default="src",
    #                     help='languge')
    # parser.add_argument('--target-lang', type=str, default="tgt",
    #                     help='sss')

    # parser.add_argument('--moses-source-lang', type=str, default="zh",
    #                     help='languge')
    # parser.add_argument('--moses-target-lang', type=str, default="zh",
    #                     help='language')
    print("返回了")

    return parser


# def check_tf_version():
#     import tensorflow as tf
#     tf_ver = tf.__version__.split('.')
#     if int(tf_ver[0]) <= 1 and int(tf_ver[1]) < 10:
#         raise ModuleNotFoundError('Tensorflow >=1.10 (one-point-ten) is required!')
#     elif int(tf_ver[0]) > 1:
#         warnings.warn('Tensorflow %s is not tested! It may or may not work. '
#                       'Feel free to submit an issue at https://github.com/hanxiao/bert-as-service/issues/' % tf.__version__)
#     return tf_ver


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def auto_bind(socket):
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'

        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')


def get_run_args(parser_fn=get_args_parser, printed=True):
    print("进来了、。。。。。")
    from bert_serving.server.page.fairseq.data import encoders
    from bert_serving.server.page.fairseq import checkpoint_utils, options, tasks, utils

    args = options.parse_args_and_arch(parser_fn())
    print("终于通过了、。。。。。")
    if printed:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


def get_benchmark_parser():

    parser = get_args_parser()

    parser.set_defaults(num_client=1, client_batch_size=4096)

    group = parser.add_argument_group('Benchmark parameters', 'config the experiments of the benchmark')

    group.add_argument('-test_client_batch_size', type=int, nargs='*', default=[1, 16, 256, 4096])
    group.add_argument('-test_max_batch_size', type=int, nargs='*', default=[8, 32, 128, 512])
    group.add_argument('-test_max_seq_len', type=int, nargs='*', default=[32, 64, 128, 256])
    group.add_argument('-test_num_client', type=int, nargs='*', default=[1, 4, 16, 64])
    group.add_argument('-test_pooling_layer', type=int, nargs='*', default=[[-j] for j in range(1, 13)])

    group.add_argument('-wait_till_ready', type=int, default=30,
                       help='seconds to wait until server is ready to serve')
    group.add_argument('-client_vocab_file', type=str, default='README.md',
                       help='file path for building client vocabulary')
    group.add_argument('-num_repeat', type=int, default=10,
                       help='number of repeats per experiment (must >2), '
                            'as the first two results are omitted for warm-up effect')
    args = options.parse_args_and_arch(parser)
    return args


def get_shutdown_parser():
    parser = argparse.ArgumentParser()
    parser.description = 'Shutting down a BertServer instance running on a specific port'

    parser.add_argument('-ip', type=str, default='localhost',
                        help='the ip address that a BertServer is running on')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, required=True,
                        help='the port that a BertServer is running on')
    parser.add_argument('-timeout', type=int, default=5000,
                        help='timeout (ms) for connecting to a server')
    return parser


class TimeContext:
    def __init__(self, msg):
        self._msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        print(self._msg, end=' ...\t', flush=True)

    def __exit__(self, typ, value, traceback):
        self.duration = time.perf_counter() - self.start
        print(colored('    [%3.3f secs]' % self.duration, 'green'), flush=True)

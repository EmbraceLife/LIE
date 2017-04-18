"""
Copyright 2016 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import signal
import atexit
import time
import sys
import argparse
import logging

from . import __version__, __homepage__
from .utils import logcolor
from . import Kurfile
from .engine import JinjaEngine

logger = logging.getLogger(__name__)

# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

###############################################################################


def parse_kurfile(filename, engine, parse=True):
    """ 1. create a Kurfile object; 2. parse the object; 4. arg1: filename with path, string; arg2: engine, as engine object; arg3: parse, boolean, default true; 5. Return: spec: parsed kurfile object
    """

    logger.info("(filename_str, engine_object, parse=True): start \n 1. create a Kurfile object, call it spec; \n 2. parse spec; \n 3. return spec; \n Inputs: \n \t 1. filename: %s; \n \t 2. engine: %s; \n \t 3. parse: default is True, %s. \n \n", filename, engine, parse)

    # initialize a Kurfile object, spec
    # see inside spec:
    # pprint(spec.__dict__.keys())
    # Note: only filename, data, engine are not None
    # pprint(spec.__dict__)
    spec = Kurfile(filename, engine)

    # parse the kurfile object spec
    # after parsing, containers is not None anymore
    if parse:
        spec.parse()

    logger.info("(filename_str, engine_object, parse=True): end \n 1. create a Kurfile object, call it spec; \n 2. parse spec; \n 3. return spec; \n Inputs: \n \t 1. filename: %s; \n \t 2. engine: %s; \n \t 3. parse: default is True, %s. \n Returns: \n \t 1. spec: type is %s, with following keys: %s \n \n", filename, engine, parse, type(spec), spec.__dict__.keys())

    return spec

###############################################################################


def dump(args):
    """ Dumps the Kurfile to stdout as a JSON blob: 1. if args.pre_parse is True, then don't parse the kurfile object; 2. then dumps the dict to print out
    """
    logger.warning("(args): print out the Kurfile details as dict on console: 1. get the kurfile object named as spec and parse it; 2. then get spec.data and print it out as a dict")

    spec = parse_kurfile(args.kurfile, args.engine, parse=not args.pre_parse)

    print(json.dumps(spec.data, sort_keys=True, indent=4))

###############################################################################


def train(args):
    """ Trains a model.
    """
    logger.warning("(args): trains a model: 1. create a kurfile object and assign info to its properties; 2. get training function for spec; 3. run this training function with args.step")

    spec = parse_kurfile(args.kurfile, args.engine)
    func = spec.get_training_function()

    func(step=args.step)

###############################################################################


def test(args):
    """ Tests a model.
    """
    spec = parse_kurfile(args.kurfile, args.engine)
    func = spec.get_testing_function()
    func(step=args.step)

###############################################################################


def evaluate(args):
    """ Evaluates a model.
    """
    spec = parse_kurfile(args.kurfile, args.engine)
    func = spec.get_evaluation_function()
    func(step=args.step)

###############################################################################


def build(args):
    """ Builds a model.
    """
    logger.warning("(args): build a model: \n 1. create a kurfile instance and fill in some of its properties; \n 2. we select from three sections: train, test, evaluate to build model; \n 3. if section is not available or args.compile as none, only do spec.get_model(provider=none) without Executor_trainer/evaluator, nor compile(); \n 4. get one or more providers from spec, but only use the default or any provider; \n 5. create a model object with spec.get_model(provider), build Executor as trainer or evaluator, and compile \n ")

    # spec = parse_kurfile(args.kurfile, args.engine)

	#####################################################
    logger.warning("step1: Initialize a Kurfile object named spec,  Now .... \n ")
    spec = Kurfile(args.kurfile, args.engine)

    # parse the kurfile object spec
    # after parsing, containers is not None anymore
    parse = True
    if parse:
        logger.warning("step2: to fill some properties of spec: Now .... \n ")
        spec.parse()

	#########################################################
    logger.warning("step3: Decide to build a bare or non-bare model on which section: Now ... \n ")
	# By default, we select all three sections: train, test, evaluate
    if args.compile == 'auto':
        result = []
        for section in ('train', 'test', 'evaluate'):
            if section in spec.data:
                result.append((section, 'data' in spec.data[section]))

		# if none of the three sections availabe in spec.data, then we don't compile model or just build a bare model; or a train section is available but has no data value, then just build a bare model
        if not result:
            logger.info('Trying to build a bare model.')
            args.compile = 'none'
        else:
            args.compile, has_data = sorted(result, key=lambda x: not x[1])[0]
            logger.info('Trying to build a "%s" model.', args.compile)
            if not has_data:
                logger.info('There is not data defined for this model, '
                            'though, so we will be running as if --bare was '
                            'specified.')

	# If args.compile set to 'none', build a bare model
    elif args.compile == 'none':
        logger.info('Trying to build a bare model.')

	# or we can select any section of the three, and build a model
    else:
        logger.info('Trying to build a "%s" model.', args.compile)

	###############################################################
    logger.warning("step4: Prepare data provider: None for bare model, default or any provider from spec: Now ... \n ")
	# if we are building a bare model, then no need for provider
    if args.bare or args.compile == 'none':
        provider = None

	# otherwise, we create a BatchProvider instance with spec (accept many data sources only when testing)
    else:
        providers = spec.get_provider(
            args.compile,
            accept_many=args.compile == 'test'
        )
		# get default provider or any provider if many providers available
        provider = Kurfile.find_default_provider(providers)


	###############################################################
    logger.warning("step5: build a model object for spec using data provider: Now ... \n ")
	# create a model object and store inside spec.model:
    spec.get_model(provider)


	###############################################################
    logger.warning("step6: Create an Executor trainer or evaluator, target: Now ... \n ")
	# if using data from train section, we build a trainer Executor and compile it; if data from test section, we build a trainer Executor without optimizer and compile it; if data from evaluate section, we build a Executor evaluator and compile it

    if args.compile == 'none':
        return
    elif args.compile == 'train':
        target = spec.get_trainer(with_optimizer=True)
    elif args.compile == 'test':
        target = spec.get_trainer(with_optimizer=False)
    elif args.compile == 'evaluate':
        target = spec.get_evaluator()
    else:
        logger.error('Unhandled compilation target: %s. This is a bug.',
                     args.compile)
        return 1

    logger.warning("step7: Compile this Executor trainer or evaluator: Now ... \n ")
	# get a backend specific representation of model
    target.compile()


###############################################################################


def prepare_data(args):
    """ Print out samples of data of a given section: 1. parse kurfile; 2. update args.target to select a section's data to look into; 3. get data from spec.data['train']['data'] dict to data supplier objects then to data provider; 4. --assemble: a. return a parsed model for spec.model, b. et trainer on Kurfile spec is to initalize a Executor object with four properties: loss, model, optimizer objects, c. compile Executor to add a compiled model with keras backend to target.model['compiled'], add real new additional_sources onto target.model['additional_sources'], d. but it seems the additional new data source is not used by BatchProvider; 5. print out a batch from data provider, or print only the first or last few samples of the batch with '--number'
    """

    logger.warning("(args): start \n Print out samples of data of a given section: \n 1. get kurfile instance spec and parse it with some info; \n 2. select a section's data to look into; \n 3. convert data from e.g., spec.data['train']['data'] dict to data supplier objects then to data provider; \n 4. --assemble: a. return a parsed model for spec.model, b. build trainer for train section is to initialize an Executor object with four properties: loss, model, optimizer objects, _, c. compile Executor is to add a compiled model using keras backend to Executor.model['compiled'], and also add real new additional_sources onto Executor.model['additional_sources'], d. but it seems the additional new data source is not used by BatchProvider (kur team will handle this in the future); \n 5. print out a batch from data provider, or print only the first or last few samples of the batch with '--number' \n \n Inputs: \n \t 1. args: \n%s \n Returns: \n \t Nothing, only print out data in consoles \n \n", args)

    # create a parsed kurfile object
    # spec is a dict, check inside spec.__dict__
    spec = parse_kurfile(args.kurfile, args.engine)

    # update args.target to be the first available section among the four
    if args.target == 'auto':
        result = None
        for section in ('train', 'validate', 'test', 'evaluate'):
            if section in spec.data and 'data' in spec.data[section]:
                result = section
                break
        if result is None:
            raise ValueError('No data sections were found in the Kurfile.')
        args.target = result

    logger.info('Preparing data sources for section: %s \n\n', args.target)

    # How data flow from external file as spec.data[section]['data'] to supplier object, to BatchProvider, and finally to a single batch?
    # return providers as a dict {'default': BatchProvider}
    providers = spec.get_provider(
        args.target,
        accept_many=args.target == 'test'
    )

    # set --assemble as True, to get additional dataset when compilation
    if args.assemble:
        # find the provider instance which is set to constructing model
		# default is BatchProvider
        default_provider = Kurfile.find_default_provider(providers)

        # return a parsed model for spec.model
		# spec.model['additional_sources'] is {} and ['compiled'] is None
        spec.get_model(default_provider)

        # get trainer on Kurfile spec is to initalize a Executor object with four properties:
        # 1. loss object; 2. Model object; 3. optimizer object; 4. auto_retry as True
        if args.target == 'train':
            target = spec.get_trainer(with_optimizer=True)
        elif args.target == 'test':
            target = spec.get_trainer(with_optimizer=False)
        elif args.target == 'evaluate':
            target = spec.get_evaluator()
        else:
            logger.error('Unhandled assembly target: %s. This is a bug.',
                         args.target)
            return 1

		# compile Executor to add a compiled model with keras backend to target.model['compiled'], add additional_sources onto target.model['additional_sources']:  ------------
        # before compile(): --
        # target.model['additional_sources'] and ['compiled'] are {} or None
        # after compile(): --
        # target.model added model.compiled: {'raw': kur.engine.training.Model}
        # in speech.yml case: model['additional_sources']:
        # {'ctc_scaled_utterance_length': <kur.loss.ctc.ScaledSource object at 0x117479f60>}
        # the problem is that this new data source is not stored in BatchProvider, not to be used by a batch
        target.compile(assemble_only=True)

    # get every data provider one at a time
    for k, provider in providers.items():
        if len(providers) > 1:
            print('Provider:', k)

        # get a batch out of the BatchProvider
        batch = None
        for batch in provider:
            break
        if batch is None:
            logger.error('No batches were produced.')
            continue

        # print the batch
        num_entries = None
        keys = sorted(batch.keys())
        num_entries = len(batch[keys[0]])

        # if --number is set, then print specific number of samples, otherwise, print a whole batch
        for entry in range(num_entries):
            if args.number is None or (args.number < num_entries and entry < args.number) or (entry - num_entries >= args.number):
                print('Entry {}/{}:'.format(entry + 1, num_entries))
                for key in keys:
                    print('  {}: {}'.format(key, batch[key][entry]))

        if num_entries is None:
            logger.error('No data sources was produced.')
            continue

    logger.warning("(args): end \n Print out samples of data of a given section: \n 1. get kurfile instance spec and parse it with some info; \n 2. select a section's data to look into; \n 3. convert data from e.g., spec.data['train']['data'] dict to data supplier objects then to data provider; \n 4. --assemble: a. return a parsed model for spec.model, b. build trainer for train section is to initialize an Executor object with four properties: loss, model, optimizer objects, _, c. compile Executor is to add a compiled model using keras backend to Executor.model['compiled'], and also add real new additional_sources onto Executor.model['additional_sources'], d. but it seems the additional new data source is not used by BatchProvider (kur team will handle this in the future); \n 5. print out a batch from data provider, or print only the first or last few samples of the batch with '--number' \n \n Inputs: \n \t 1. args: \n%s \n Returns: \n \t Nothing, only print out data in consoles \n \n ", args)

###############################################################################


def version(args):							# pylint: disable=unused-argument
    """ Prints the Kur version and exits.
    """
    print('Kur, by Deepgram -- deep learning made easy')
    print('Version: {}'.format(__version__))
    print('Homepage: {}'.format(__homepage__))

###############################################################################


def do_monitor(args):
    """ Handle "monitor" mode.
    """

    # If we aren't running in monitor mode, then we are done.
    if not args.monitor:
        return

    # This is the main retry loop.
    while True:
        # Fork the process.
        logger.info('Forking child process.')
        pid = os.fork()

        # If we are the child, leave this function and work.
        if pid == 0:
            logger.info('We are a newly spawned child process.')
            return

        logger.info('Child process spawned: %d', pid)

        # Wait for the child to die. If we die first, kill the child.
        atexit.register(kill_process, pid)
        try:
            _, exit_status = os.waitpid(pid, 0)
        except KeyboardInterrupt:
            break
        atexit.unregister(kill_process)

        # Process the exit code.
        signal_number = exit_status & 0xFF
        exit_code = (exit_status >> 8) & 0xFF
        core_dump = bool(0x80 & signal_number)

        if signal_number == 0:
            logger.info('Child process exited with exit code: %d.', exit_code)
        else:
            logger.info('Child process exited with signal %d (core dump: %s).',
                        signal_number, core_dump)

        retry = False
        if os.WIFSIGNALED(exit_status):
            if os.WTERMSIG(exit_status) == signal.SIGSEGV:
                logger.error('Child process seg faulted.')
                retry = True

        if not retry:
            break

    sys.exit(0)

###############################################################################


def kill_process(pid):
    """ Kills a child process by PID.
    """

    # Maximum time we wait (in seconds) before we send SIGKILL.
    max_timeout = 60

    # Terminate child process
    logger.info('Sending Ctrl+C to the child process %d', pid)
    os.kill(pid, signal.SIGINT)

    start = time.time()
    while True:
        now = time.time()

        # Check the result.
        result = os.waitpid(pid, os.WNOHANG)
        if result != (0, 0):
            # The child process is dead.
            break

        # Check the timeout.
        if now - start > max_timeout:
            # We've waited too long.
            os.kill(pid, signal.SIGKILL)
            break

        # Keep waiting.
        logger.debug('Waiting patiently...')
        time.sleep(0.5)

###############################################################################


def parse_args():
    """ Create console args for kur program: 1. Constructs an argument parser and the framework; 2. create parser args like '--no-color', '--verbose', '--monitor', '--version'; 3. create cmd subparsers like 'train', 'evaluate', 'test', 'build', 'dump', 'data'; 4. create subparse args like '--step', '--compile' with choices, '--bare', '--pre-parse', '--target' with choices, '--assemble', '--number' with integer; 5. return this parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description='Descriptive deep learning')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colorful logging.')
    parser.add_argument('-v', '--verbose', default=0, action='count',
                        help='Increase verbosity. Can be specified twice for debug-level '
                        'output.')
    parser.add_argument('--monitor', action='store_true',
                        help='Run Kur in monitor mode, which tries to recover from critical '
                        'errors, like segmentation faults.')
    parser.add_argument('--version', action='store_true',
                        help='Display version and exit.')

    subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

    subparser = subparsers.add_parser('train', help='Trains a model.')
    subparser.add_argument('--step', action='store_true',
                           help='Interactive debug; prompt user before submitting each batch.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.set_defaults(func=train)

    subparser = subparsers.add_parser('test', help='Tests a model.')
    subparser.add_argument('--step', action='store_true',
                           help='Interactive debug; prompt user before submitting each batch.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.set_defaults(func=test)

    subparser = subparsers.add_parser('evaluate', help='Evaluates a model.')
    subparser.add_argument('--step', action='store_true',
                           help='Interactive debug; prompt user before submitting each batch.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.set_defaults(func=evaluate)

    subparser = subparsers.add_parser('build',
                                      help='Tries to build a model. This is useful for debugging a model.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.add_argument('-c', '--compile',
                           choices=['none', 'train', 'test', 'evaluate', 'auto'], default='auto',
                           help='Try to compile the specified variation of the model. If '
                           '--compile=none, then it only tries to assemble the model, not '
                           'compile anything. --compile=none implies --bare')
    subparser.add_argument('-b', '--bare', action='store_true',
                           help='Do not attempt to load the data providers. In order for your '
                           'model to build correctly with this option, you will need to '
                           'specify shapes for all of your inputs.')
    subparser.set_defaults(func=build)

    subparser = subparsers.add_parser('dump',
                                      help='Dumps the Kurfile out as a JSON blob. Useful for debugging.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.add_argument('-p', '--pre-parse', action='store_true',
                           help='Dump the Kurfile before parsing it.')
    subparser.set_defaults(func=dump)

    subparser = subparsers.add_parser('data',
                                      help='Does not actually compile anything, but only prints out a '
                                      'single batch of data. This is useful for debugging data sources.')
    subparser.add_argument('kurfile', help='The Kurfile to use.')
    subparser.add_argument('-t', '--target',
                           choices=['train', 'validate', 'test', 'evaluate', 'auto'],
                           default='auto', help='Try to produce data corresponding to a specific '
                           'variation of the model.')
    subparser.add_argument('--assemble', action='store_true', help='Also '
                           'begin assembling the model to pull in compile-time, auxiliary data '
                           'sources.')
    subparser.add_argument('-n', '--number', type=int, help='number of samples to print')
    subparser.set_defaults(func=prepare_data)

	# note: what is the output
    return parser.parse_args()

###############################################################################


def main():
    """
    "main(): 1. get console args into program; 2. configurate logging display; 3. monitor process when required by args; 4. show version or do nothing when required by args; 5. create an JinjaEngine object, and assign it to args.engine; 6. run args.func(args) and then exit program. There are many functions to try: 1. !kur data | dump | build | train | test | evaluate"
    """

    # get console arguments
    args = parse_args()


    # configurate logging display
    loglevel = {
        0: logging.WARNING,
        1: logging.CRITICAL,
        2: logging.INFO,
        3: logging.DEBUG

    }
    config = logging.basicConfig if args.no_color else logcolor.basicConfig
    config(
        level=loglevel.get(args.verbose, logging.DEBUG),
        format='{color}[%(levelname)s %(asctime)s %(name)s %(funcName)s:%(lineno)s]{reset} '
        '%(message)s'.format(
            color='' if args.no_color else '$COLOR',
            reset='' if args.no_color else '$RESET'
        )
    )
    logging.captureWarnings(True)


    logger.warning("console args: \n \t %s \n", args)


    # monitor process when required by args
    do_monitor(args)

    # show version or do nothing when required by args
    if args.version:
        args.func = version
    elif not hasattr(args, 'func'):
        print('Nothing to do!', file=sys.stderr)
        print('For usage information, try: kur --help', file=sys.stderr)
        print('Or visit our homepage: {}'.format(__homepage__))
        sys.exit(1)

    # create an JinjaEngine object, and assign it to args.engine
    engine = JinjaEngine()
    setattr(args, 'engine', engine)

    logger.warning("(): \n 1. get console args into program; \n 2. configurate logging display; \n 3. monitor process when required by args; \n 4. show version or do nothing when required by args; \n 5. create an JinjaEngine object, and assign it to args.engine; \n 6. run args.func(args) and then exit program. \n There are many functions to try: !kur data | dump | build | train | test | evaluate \n \n the last function to run before exit program: %s(args) \n \n", args.func)
    # run args.func(args) and then exit program
    sys.exit(args.func(args) or 0)


###############################################################################
if __name__ == '__main__':
    main()

# EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

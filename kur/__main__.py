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
import numpy as np
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

###############################################################################
def parse_kurfile(filename, engine, parse=True):
	""" Parses a Kurfile.

		# Arguments

		filename: str. The path to the Kurfile to load.

		# Return value

		Kurfile instance
	"""
	logger.info("(filename_str, engine_object, parse=True): start \n 1. create a Kurfile object, call it spec; \n 2. parse spec; \n 3. return spec; \n Inputs: \n1. filename: %s; \n2. engine: %s; \n3. parse: default is True, %s. \n\n", filename, engine, parse)

	spec = Kurfile(filename, engine)
	if parse:
		spec.parse()

	logger.info("(filename_str, engine_object, parse=True): end \n1. create a Kurfile object, call it spec; \n2. parse spec; \n3. return spec; \nInputs: \n1. filename: %s; \n2. engine: %s; \n3. parse: default is True, %s. \n\nReturns: \n1. spec: type is %s \n2. spec keys are: \n%s \n\n", filename, engine, parse, type(spec), spec.__dict__.keys())

	return spec

###############################################################################
def dump(args):
	""" Dumps the Kurfile to stdout as a JSON blob.
	"""
	logger.warning("(args): \nprint out the Kurfile details as dict on console: \n1. get the kurfile object named as spec and parse it; \n2. then get spec.data and print it out as a dict \n\n")

	spec = parse_kurfile(args.kurfile, args.engine, parse=not args.pre_parse)
	print(json.dumps(spec.data, sort_keys=True, indent=4))

###############################################################################
def train(args):
	""" Trains a model.
	"""
	logger.warning("(args): trains a model \n1. create a kurfile object and assign info to its properties; \n2. get training function for spec; \n3. run this training function with args.step \n\n")

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
	logger.warning("(args): build a model: \n1. create a kurfile instance and fill in some of its properties; \n2. we select from three sections: train, test, evaluate to build model; \n3. if section is not available or args.compile as none, only do spec.get_model(provider=none) without Executor_trainer/evaluator, nor compile(); \n4. get one or more providers from spec, but only use the default or any provider; \n5. create a model object with spec.get_model(provider), build Executor as trainer or evaluator, and compile \n\n ")


	spec = parse_kurfile(args.kurfile, args.engine)

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
				logger.warning('There is not data defined for this model, '
					'so we will be running as if --bare was specified.')
	# If args.compile set to 'none', build a bare model
	elif args.compile == 'none':
		logger.info('Trying to build a bare model.')
	# or we can select any section of the three, and build a model
	else:
		logger.info('Trying to build a "%s" model.', args.compile)

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

	# create a model object and store inside spec.model:
	spec.get_model(provider)

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
	# get a backend specific representation of model
	target.compile()

###############################################################################
def prepare_data(args):
	""" Prepares a model's data provider.
	"""

	logger.warning("(args): start \n Print out samples of data of a given section: \n 1. get kurfile instance spec and parse it with some info; \n 2. select a section's data to look into; \n 3. convert data from e.g., spec.data['train']['data'] dict to data supplier objects then to data provider; \n 4. --assemble: a. return a parsed model for spec.model, b. build trainer for train section is to initialize an Executor object with four properties: loss, model, optimizer objects, _, c. compile Executor is to add a compiled model using keras backend to Executor.model['compiled'], and also add real new additional_sources onto Executor.model['additional_sources'], d. but it seems the additional new data source is not used by BatchProvider (kur team will handle this in the future); \n 5. print out a batch from data provider, or print only the first or last few samples of the batch with '--number' \n \n Inputs: \n \t 1. args: \n%s \n Returns: \n \t Nothing, only print out data in consoles \n \n", args)

	logger.warning("step1: \nCreate a Kurfile object and parse it with detailed information \n\n")

	#original
	spec = parse_kurfile(args.kurfile, args.engine)

	logger.critical("\n The parsed kurfile object conatins the following dicts: \n\n")
	for k in spec.__dict__:
		print(k, "\n")
		pprint(spec.__dict__[k])
		print("\n\n")



	logger.warning("step2: \nSelect the first available section of ('train', 'validate', 'test', 'evaluate'); \n\n")

	if args.target == 'auto':
		result = None
		for section in ('train', 'validate', 'test', 'evaluate'):
			if section in spec.data and 'data' in spec.data[section]:
				result = section
				break
		if result is None:
			raise ValueError('No data sections were found in the Kurfile.')
		args.target = result

	logger.critical("In this case, the selected section: {}\n\n".format(args.target))

	logger.warning("step3: \nGet data provider for the section \n\n")

	providers = spec.get_provider(
		args.target,
		accept_many=args.target == 'test'
	)

	logger.critical("\nThe data provider contains: \n\n")
	pprint(providers)
	print("\n\n")
	logger.critical("\nThe batch provider contains: \n\n")
	pprint(providers['default'].__dict__)
	print("\n\n")
	logger.critical("\nCheck inside each data source of data provider: \n\n")
	for source in providers['default'].__dict__['sources']:
		print(source, ": has the following content \n")
		# pprint(source.__dict__)
		for k, v in source.__dict__.items():
			if isinstance(v, list):
				print("Key:", k, "; Value : is a list of", len(v), "num of ", type(v[0]))
				print("print v[0:5]")
				pprint(v[0:5])
			elif isinstance(v, np.ndarray):
				print("Key: ", k, "; Value : is array of shape ", v.shape)
				if len(v.shape)>1:
					print("print v[0:1]")
					pprint(v[0:1])
				else:
					print("print v[0:5]")
					pprint(v[0:5])
			else:
				print(k, ":", v)
		print("\n\n")

	logger.warning("step4: \nIf args.assemble == True, do the following: \n1. get default data provider; \n2. get a parsed model for spec.model; \n3. build trainer for train section is to initialize an Executor object with four properties: loss, model, optimizer objects... \n4. compile Executor is to add a compiled model using keras backend to Executor.model['compiled'], and also add real new additional_sources onto Executor.model['additional_sources']\n\n")

	if args.assemble:
		default_provider = Kurfile.find_default_provider(providers)

		logger.critical("\n\nFind the provider object for constructing the model here: \n%s \n\n", default_provider)

		spec.get_model(default_provider)

		logger.critical("\nspec.model is updated, not None any more: \n%s \n\n", spec.model)
		pprint(spec.model.__dict__)
		print("\n\n")

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

		logger.critical("\nCreate Executor trainer using spec: is to create a large dict with loss, model, optimizer \n%s \n\n", target)
		pprint(target.__dict__)
		print("\n\n")

		target.compile(assemble_only=True)

		logger.critical("\nCompile Executor trainer above \n%s  \nTo compile is to create  `target.model.compiled['raw'].__dict__` with a keras backend Model object \n\nThis keras-backend Model object is \n", target)
		pprint(pprint(target.model.compiled['raw'].__dict__))
		print("\n\n")

	logger.warning("step5: \nPrint out data samples \n1. Get (every) data provider \n2. get just one batch from this provider; \n3. print the whole batch; \n4. print specific number of data samples using args.number \n\n")

	for k, provider in providers.items():
		if len(providers) > 1:
			print('Provider:', k)

		batch = None
		for batch in provider:
			break
		if batch is None:
			logger.error('No batches were produced.')
			continue

		logger.critical("\nWhat a provider look like? \n\n")
		pprint(provider.__dict__)
		print("\n\n")

		logger.critical("\nWhat a batch directly from provider look like? batch type: %s\n\n", type(batch))

		for k, v in batch.items():
			print(k, " : ", type(v), v.shape)
		print("\n\n")

		num_entries = None
		keys = sorted(batch.keys())
		num_entries = len(batch[keys[0]])
		for entry in range(num_entries):
			if args.number is None or entry < args.number or \
				(entry - num_entries >= args.number):
				print('Entry {}/{}:'.format(entry+1, num_entries))
				for key in keys:
					print('  {}: {}'.format(key, batch[key][entry]))

		if num_entries is None:
			logger.error('No data sources was produced.')
			continue


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
			logger.debug('We are a newly spawned child process.')
			return

		logger.debug('Child process spawned: %d', pid)

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
	logger.debug('Sending Ctrl+C to the child process %d', pid)
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
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description='Descriptive deep learning')
	parser.add_argument('--no-color', action='store_true',
		help='Disable colorful logging.')
	parser.add_argument('-v', '--verbose', default=0, action='count',
		help='Increase verbosity. Can be specified up to three times for '
			'trace-level output.')
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
	subparser.add_argument('-n', '--number', type=int,
		help='Number of samples to print (default: the entire batch).')
	subparser.set_defaults(func=prepare_data)

	return parser.parse_args()

###############################################################################
def main():
	""" Entry point for the Kur command-line script.
	"""
	args = parse_args()

	loglevel = {
		0 : logging.WARNING,
		1 : logging.CRITICAL,
		2 : logging.INFO,
		3 : logging.DEBUG,
		4 : logging.TRACE
	}
	config = logging.basicConfig if args.no_color else logcolor.basicConfig
	config(
		level=loglevel.get(args.verbose, logging.TRACE),
		format='{color}[%(levelname)s %(asctime)s %(name)s %(funcName)s:%(lineno)s]{reset} '
			'%(message)s'.format(
				color='' if args.no_color else '$COLOR',
				reset='' if args.no_color else '$RESET'
			)
	)
	logging.captureWarnings(True)

	# print args at beginning of every run
	logger.warning("console args: \n \t %s \n", args)

	do_monitor(args)

	if args.version:
		args.func = version
	elif not hasattr(args, 'func'):
		print('Nothing to do!', file=sys.stderr)
		print('For usage information, try: kur --help', file=sys.stderr)
		print('Or visit our homepage: {}'.format(__homepage__))
		sys.exit(1)

	engine = JinjaEngine()
	setattr(args, 'engine', engine)

	logger.warning("(): \n1. get console args into program; \n2. configurate logging display; \n3. monitor process when required by args; \n4. show version or do nothing when required by args; \n5. create an JinjaEngine object, and assign it to args.engine; \n6. run args.func(args) and then exit program. \n\nThere are many functions to try: !kur data | dump | build | train | test | evaluate \n\nRun %s(args) before exit program  \n\n", args.func)

	sys.exit(args.func(args) or 0)

###############################################################################
if __name__ == '__main__':
	main()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

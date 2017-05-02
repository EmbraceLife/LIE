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

from . import __version__, __homepage__
from .utils import logcolor
from . import Kurfile
from .engine import JinjaEngine

import logging
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
from .utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
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
	spec = Kurfile(filename, engine)
	if parse:
		spec.parse()
	return spec

###############################################################################
def dump(args):
	""" Dumps the Kurfile to stdout as a JSON blob.
	"""
	spec = parse_kurfile(args.kurfile, args.engine, parse=not args.pre_parse)
	print(json.dumps(spec.data, sort_keys=True, indent=4))

###############################################################################
def train(args):
	""" Trains a model.
	"""
	logger.critical("\n\nspec = parse_kurfile(args.kurfile, args.engine)  \n\n")
	spec = parse_kurfile(args.kurfile, args.engine)

	logger.critical("\n\nfunc = spec.get_training_function()  \n\nThen run `func(step=args.step)`\n\nEOF \n\nDive inside `func = spec.get_training_function()`\n\n")
	func = spec.get_training_function()

	# logger.critical("\n\nfunc(step=args.step)\n\nEOF \n\nDive into func() inside get_training_function\n\n")
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
	logger.critical("\n\nspec = parse_kurfile(args.kurfile, args.engine)\n\n")
	spec = parse_kurfile(args.kurfile, args.engine)

	func_provider = """
	providers = spec.get_provider(
		args.compile,
		# args.compile = "auto", "train", "test", "evalute", "none"
		accept_many=args.compile == 'test'
	)
	provider = Kurfile.find_default_provider(providers)
	"""

	logger.critical("\n\nget data provider from a section(mostly train, or any specified section)\n\n%s\n\n", func_provider)

	if args.compile == 'auto':
		result = []
		for section in ('train', 'test', 'evaluate'):
			if section in spec.data:
				result.append((section, 'data' in spec.data[section]))
		if not result:
			logger.info('Trying to build a bare model.')
			args.compile = 'none'
		else:
			args.compile, has_data = sorted(result, key=lambda x: not x[1])[0]
			logger.debug('Trying to build a "%s" model.', args.compile)
			if not has_data:
				logger.warning('There is not data defined for this model, '
					'so we will be running as if --bare was specified.')
	elif args.compile == 'none':
		logger.info('Trying to build a bare model.')
	else:
		logger.info('Trying to build a "%s" model.', args.compile)

	if args.bare or args.compile == 'none':
		provider = None
	else:
		providers = spec.get_provider(
			args.compile,
			accept_many=args.compile == 'test'
		)
		provider = Kurfile.find_default_provider(providers)

	logger.critical("\n\nCreate, parse, build kur.Model object using \n\nspec.get_model(provider) \n\n")

	spec.get_model(provider)




	if args.compile == 'none':
		logger.critical("\n\nargs.compile == 'none', then return with Nothing\n\n")
		return
	elif args.compile == 'train':
		func_str = """
return Executor(
	model=self.get_model(),
	loss=self.get_loss(),
	optimizer=self.get_optimizer() if with_optimizer else None
		)
		"""

		target = spec.get_trainer(with_optimizer=True)

		logger.critical("\n\nMain purpose of Executor object is to put loss, optimizer, model together in one place for this model, using\n\ntarget = spec.get_trainer(with_optimizer=True)\n\nCreate an Executor with model, loss and optimizer of this spec, using\n\n%s\n\n", func_str)

		logger.warning("\n\nLet's see inside the Executor object for %s\n\n", args.compile)
		pprint(target.__dict__)
		print("\n\n")

	elif args.compile == 'test':
		logger.critical("\n\nMain purpose of Executor object is to put loss, optimizer, model together in one place for this model, using\n\ntarget = spec.get_trainer(with_optimizer=False)\n\nCreate an Executor with model, loss and optimizer of this spec, using\n\n%s\n\nLet's see inside the Executor object for %s\n\n", func_str, args.compile)
		target = spec.get_trainer(with_optimizer=False)
		pprint(target.__dict__)
		print("\n\n")
	elif args.compile == 'evaluate':
		logger.critical("\n\nMain purpose of Executor object is to put loss, optimizer, model together in one place for this model, using\n\ntarget = spec.get_evaluator()\n\nCreate an Executor with model, loss and optimizer of this spec, using\n\n%s\n\nLet's see inside the Executor object for %s\n\n", func_str, args.compile)
		target = spec.get_evaluator()
		pprint(target.__dict__)
		print("\n\n")
	else:
		logger.error('Unhandled compilation target: %s. This is a bug.',
			args.compile)
		return 1

	logger.critical("\n\nUse target.compile() to compile model object\n\nTo create spec.model.compiled[key] and spec.model.compiled['raw']\n\nThen save initial weights from compiled['raw'] into external idx files, \ntest func from compiled[key] onto compiled['raw'], \nrestore initial weights back to variables in model\n\nEOF\n\n")

	target.compile()

###############################################################################
def prepare_data(args):
	""" Prepares a model's data provider.
	"""

	logger.critical("\n\nCreate a Kurfile object and parse it with detailed information \n\n")

	spec = parse_kurfile(args.kurfile, args.engine)

	logger.critical("\n\nSelect a section to take its data provider\n\n")
	if args.target == 'auto':
		result = None
		for section in ('train', 'validate', 'test', 'evaluate'):
			if section in spec.data and 'data' in spec.data[section]:
				result = section
				break
		if result is None:
			raise ValueError('No data sections were found in the Kurfile.')
		args.target = result

	logger.critical('The section for data provider is: %s', args.target)

	logger.critical("\n\nDive into how data is flowing from external fines into a data provider\n\ndata provider is ready to iterate out batches one by one\n\n")
	providers = spec.get_provider(
		args.target,
		accept_many=args.target == 'test'
	)

	logger.critical("\n\nIf assemble is required, then \n\nspec.get_model(default_provider); \n\ntarget = spec.get_trainer(with_optimizer=True); \n\ntarget.compile(assemble_only=True)\n\n")


	if args.assemble:

		default_provider = Kurfile.find_default_provider(providers)
		spec.get_model(default_provider)

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

		target.compile(assemble_only=True)

	logger.critical("\n\nGet a batch out of the data provider, print it out\n\nOr just print out first or last few samples\n\n")
	for k, provider in providers.items():
		if len(providers) > 1:
			print('Provider:', k)

		batch = None
		for batch in provider:
			break
		if batch is None:
			logger.error('No batches were produced.')
			continue

		for k, v in batch.items():
			print("See a batch's keys and shapes: \n")
			print("key:", k)
			print("shape:", v.shape)

		num_entries = None
		keys = sorted(batch.keys())
		num_entries = len(batch[keys[0]])
		for entry in range(num_entries):
			if args.number is None or entry < args.number or \
				(entry - num_entries >= args.number):
				print('Entry {}/{}:'.format(entry+1, num_entries))
				for key in keys:
					print('  {}: {}'.format(key, batch[key][entry]))


		# plotting a few images
		for k, v in spec.data['train']['data'][0].items():
			name1 = k

		if name1 == 'mnist':
			images_p = batch['images'][0:9]
			image_dim = images_p.shape[1:-1]

		elif name1 == 'cifar':
			images_p = batch['images'][0:9]
			image_dim = images_p.shape[1:]

		else:
			return None # no plotting

		labels_p = [np.argmax(label) for label in batch['labels'][0:9]]

		plot_images(images=images_p, cls_true=labels_p, image_dim=image_dim)

		if num_entries is None:
			logger.error('No data sources was produced.')
			continue

# borrowed from https://hyp.is/9KkYnCyIEeeeUt-8LW6lMw/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb
def plot_images(images, cls_true, image_dim, cls_pred=None):

	assert len(images) == len(cls_true) == 9

	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.
		ax.imshow(images[i].reshape(image_dim), cmap='binary')


		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()
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
		0 : logging.CRITICAL,
		1 : logging.WARNING,
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

	logger.critical("\n\nstart main()\n\n")

	logger.critical("\n\nargs = parse_args()\n\n")
	pprint(args)
	print("\n\n")

	logger.critical("\n\nconfigurate logging system\n\n")

	logger.critical("\n\ndo_monitor(args)\n\n")
	do_monitor(args)

	logger.critical("\n\nget version or do Nothing\n\n")
	if args.version:
		args.func = version
	elif not hasattr(args, 'func'):
		print('Nothing to do!', file=sys.stderr)
		print('For usage information, try: kur --help', file=sys.stderr)
		print('Or visit our homepage: {}'.format(__homepage__))
		sys.exit(1)

	logger.critical("\n\nengine = JinjaEngine()\n\nsetattr(args, 'engine', engine)\n\n")
	engine = JinjaEngine()
	setattr(args, 'engine', engine)

	logger.critical("\n\nsys.exit(args.func)(args) or 0)\n\nEOF\n\n")
	print("Dive into __main__.{}\n\n".format(args.func))


	sys.exit(args.func(args) or 0)

###############################################################################
if __name__ == '__main__':
	main()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

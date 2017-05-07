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
from .supplier import Supplier
from .providers import BatchProvider
from .plugins import Plugin
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
	logger.critical("\n\n1. get kurfile object, using\n\n`spec = parse_kurfile(args.kurfile, args.engine)`\n\n2. select a section for data provider, or no section no data provider, using\n\n`providers = spec.get_provider(args.compile,accept_many=args.compile == 'test')`\n`provider = Kurfile.find_default_provider(providers)`\n\n3. create model and assigned to `spec` using \n\n`spec.get_model(provider)`\n\n4. create Executor trainer for train section (with optimizer), or test section (without optimizer), or create Executor evaluator  for evaluate section (without loss, optimizer), using \n\n`target = spec.get_trainer(with_optimizer=True)`\n\n`target = spec.get_evaluator()`\n\n")

	if logger.isEnabledFor(logging.CRITICAL):
		print("4. Use target.compile() to compile model object in specific backend lib\n\n4.1. create spec.model.compiled[key] and spec.model.compiled['raw']\n4.2. Then save initial weights from compiled['raw'] into external idx files, \n4.3. test func using compiled[key] onto compiled['raw'] on a provider with just 2 samples\n4.4. restore initial weights back to variables in model\n\nEOF\n\n")

	# step1
	logger.critical("\n\nStep1. Create a Kurfile object, using\n\n`Kurfile.__init__`\n\nand parse it, using\n\n`Kurfile.parse`\n\n")

	spec = parse_kurfile(args.kurfile, args.engine)



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

	logger.critical("\n\nStep2. select a section using args.compile\n\nargs.compile can be set as 'auto' by default (train, or test, or evaluate), \n\nor none (without data provider), \n\nor set as a specific section 'train' or 'test' or 'evaluate')\n\nIn this case, we build with section: %s", args.compile)



	logger.critical("\n\nStep3. get data provider of the section, or none if no section selected\n\nAccess the data provider of the selected section above, using \n\n`providers = spec.get_model(args.compile, accept_many=args.compile == 'test')`\n`provider = Kurfile.find_default_provider(providers)` \n\n")

	if args.bare or args.compile == 'none':
		provider = None
	else:
		providers = spec.get_provider(
			args.compile,
			accept_many=args.compile == 'test'
		)
		provider = Kurfile.find_default_provider(providers)



	# step3. get model and get Executor for trainer,
	logger.critical("\n\nStep4. Create model and assigned to spec\n\nDive into `Kurfile.get_model`, then into `Model.__init__`, `Model.parse`, `Model.register_provider`, `Model.build`\n\n")

	spec.get_model(provider)



	# step4. get Executor trainer or evaluator
	logger.critical("\n\nStep5. get Executor trainer or evaluator is to prepare loss, optimizer and model in one place\n\nDive into `Kurfile.get_trainer` or `Kurfile.get_evaluator`, \n\nthen into `Executor.__init__`\n\n")

	if args.compile == 'none':
		logger.critical("\n\nargs.compile == 'none', then return with Nothing\n\n")
		return
	elif args.compile == 'train':

		target = spec.get_trainer(with_optimizer=True)

		logger.critical("\n\nExecutor for train section: \n\n")
		if logger.isEnabledFor(logging.CRITICAL):
			pprint(target.__dict__)
			print("\n\n")

	elif args.compile == 'test':

		target = spec.get_trainer(with_optimizer=False)
		logger.critical("\n\nExecutor for test section: \n\n")
		if logger.isEnabledFor(logging.CRITICAL):
			pprint(target.__dict__)
			print("\n\n")

	elif args.compile == 'evaluate':

		target = spec.get_evaluator()
		logger.critical("\n\nExecutor for evaluate section: \n\n")
		if logger.isEnabledFor(logging.CRITICAL):
			pprint(target.__dict__)
			print("\n\n")

	else:
		logger.error('Unhandled compilation target: %s. This is a bug.',
			args.compile)
		return 1

	logger.critical("\n\nStep6. Compile model\n\n1. To create spec.model.compiled[key] and spec.model.compiled['raw']\n\n2. Then save initial weights from compiled['raw'] into external idx files, \n\n3. test func using compiled[key] onto compiled['raw'] on a provider with just 2 samples\n\n4. restore initial weights back to variables in model\n\nEOF\n\n")

	target.compile()
	# Aboe is `kur build` ####################


	# # Experiment start here  ############################
	# # a batch is ready from above
	# # get weights from a particular weight file
	#
	# # create a single data sample
	# sample = None
	# for batch in provider:
	# 	sample = batch['images'][0]
	# 	break
	#
	# # make weights and biases available as numpy arrays
	# import kur.utils.idx as idx
	# dense_w = idx.load("../../Hvass_tutorial1_folders/mnist.best.valid.w/layer___dense_0+weight.kur")
	# dense_b = idx.load("../../Hvass_tutorial1_folders/mnist.best.valid.w/layer___dense_0+bias.kur")
	#
	# # make activation object available
	# set_trace()
	# import keras.backend.tensorflow_backend as K
	# # data_f = K.flatten(batch_images[0])
	# data_flat = batch_images[0].reshape((1, -1))
	# data_fw = data_flat.dot(dense_w.T)
	# data_fw_v = data_fw.reshape((-1,))
	# dense_h = data_fw_v + dense_b
	# # the problem is I don't know the operators at all
	# set_trace()
	# #################################

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

	logger.critical("\n\nDive into how data is flowing from external fines into a data provider\n\ndata provider is ready to iterate out batches one by one\n\nDive into `providers = spec.get_provider(args.target, accept_many=args.target == 'test'`\n\n")


	logger.critical("Kurfile.get_provider( section, accept_many=False):  \n\nUsing detailed info from spec.data[section]['data'] to build data suppliers first, then build a data provider : \n\n1. get the dict of spec.data[section]; \n\n2. make sure spec.data[section] has a key as 'data' or 'provider'; \n\n3. store spec.data[section]['data'] in 'supplier_list', then make it a dict with key 'default'; \n\n4. Make sure there is no more than one data sources execpt when accept_many=True; \n\n5. create data Supplier object from spec.data[section]['data']; \n\n6. create data provider using data supplier and provider_detailed_info from spec.data[section]['provider']; \n\n7. finally return this provider\n\n")

	logger.critical("\n\nLook at each step above in details: \n\n")
	if logger.isEnabledFor(logging.CRITICAL):
		print("1. get the dict of spec.data[section];\n\n2. make sure spec.data[section] has a key as 'data' or 'provider';\n\nSee spec.data['train']['data'] below, note: it is a list of dict\n")
		pprint(spec.data[args.target]['data'])
		print("\n\n")
		print("3. store `spec.data[section]['data']` in 'supplier_list', then make it a dict with key 'default';\n\n `supplier_list['default'] = spec.data[section]['data']`\n\n4. Make sure there is no more than one data sources execpt when accept_many=True;\n\n")
		print("5. create data Supplier object from spec.data[section]['data']; \n")
		print("""
suppliers = {}
for k, v in supplier_list.items():
	if not isinstance(v, (list, tuple)):
		raise ValueError('Data suppliers must form a list of '
			'suppliers.')
	suppliers[k] = [
		Supplier.from_specification(entry, kurfile=self)
		for entry in v
	]

For simplicity, we use:
# Note: entry_supplier, must be a dict, rather than a list of dict

entry_supplier = spec.data[args.target]['data'][0]
data_supplier = Supplier.from_specification(entry_supplier, kurfile=spec)

		""")

		print("\nLet's dive inside this data_supplier\n")
		entry_supplier = spec.data[args.target]['data'][0]
		data_supplier = Supplier.from_specification(entry_supplier, kurfile=spec)
		pprint(data_supplier.__dict__)
		print("\n\n")

		print("5.5 How exactly entry_supplier become data_supplier object? \n\nFirst, extract a Supplier Class name in this entry_supplier dict, see `name` below, note: entry_supplier == spec below\n")
		print("""
candidates = set(
	cls.get_name() for cls in Supplier.get_all_suppliers()
) & set(spec.keys())
name = candidates.pop()
		""")
		candidates = set(
			cls.get_name() for cls in Supplier.get_all_suppliers()
		) & set(entry_supplier.keys())
		name = candidates.pop()
		print("\nthe above, name: {}".format(name))
		print("\nThen extract data_details as params and try to get supplier_name if available\n")
		print("""
params = spec[name]

# maybe, the kurfile added a dict = name: supplier_class
supplier_name = spec.get('name')
		""")
		params = entry_supplier[name]
		supplier_name = entry_supplier.get('name')
		print("\nparams: {}\nsupplier_name: {}\n".format(params, supplier_name))
		print("\nFinally, we get the specific Supplier class, and instantiate its object\n\n")
		print("""
if isinstance(params, dict):

	# Note: pay attention to the argument inputs for MnistSupplier

	result = Supplier.get_supplier_by_name(name)(
		name=supplier_name, kurfile=spec, **params)

...
		""")
		print("\nAbove function takes 2 steps: \n\nstep1: Supplier.__init__ \n\n")
		print("""
def __init__(self, name=None, kurfile=None):
	# Creates a new supplier.

	self.name = name
	self.kurfile = kurfile
		""")
		print("\nStep2: MnistSupplier.__init__: \n\n")
		print("""

# Attention: labels, images can be hidden inside **params
# Attention: name and kurfile can be given as args to MnistSupplier

def __init__(self, labels, images, *args, **kwargs):
	super().__init__(*args, **kwargs)

# Attention: MnistSupplier.methods are doing a lot of useful preprocessing here

	self.data = {
		'images' : MnistSupplier._normalize(
			VanillaSource(idx.load(MnistSupplier._get_filename(images)))
		),
		'labels' : MnistSupplier._onehot(
			VanillaSource(idx.load(MnistSupplier._get_filename(labels)))
		)
}
		""")


		print("6. Create provider object using supplier object and provider_spec\n")

		print("Get provider_detailed_info: to be extracted from  spec.data['train']['provider']\n")
		provider_spec = spec.data['train']['provider']
		pprint(provider_spec)
		print("\n\n")
		print("As there is no provider_spec['name'], we use default provider: BatchProvider\n\n`provider = Kurfile.DEFAULT_PROVIDER = BatchProvider`\n\n")

		print("Now, create provider from supplier and provider_spec\n")
		print("""
provider = BatchProvider
p = {}

Msupplier = {}
Msupplier['default'] = [data_supplier]

# note: v must be a list of dict
for k, v in Msupplier.items():

	p[k] = provider(
		sources=Supplier.merge_suppliers(v),
		**provider_spec
	)
pprint(p)
# Note: p is iterable to throw out batches
		""")
		provider = BatchProvider
		p = {}

		Msupplier = {}
		Msupplier['default'] = [data_supplier]

		for k, v in Msupplier.items():

			p[k] = provider(
				sources=Supplier.merge_suppliers(v),
				**provider_spec
			)
		pprint(p)


	providers = spec.get_provider(
		args.target,
		accept_many=args.target == 'test'
	)



	logger.critical("\n\nIf assemble is required, then \n\nspec.get_model(default_provider); \n\ntarget = spec.get_trainer(with_optimizer=True); \n\ntarget.compile(assemble_only=True)\n\nDon't set assemble=True, let `build` takes care of Model and Compilation\n\n")


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

		# take a batch of data out of provider
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


		# print out a batch or a few samples of data (images, labels)
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
		if args.plot_images:
			logger.critical("\n\nPlot 9 images\n\n")
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
def why(args):
	""" Print out reasons for using kur primarily over other libraries.
	"""
	import webbrowser
	import time
	print("See reasons of why I use kur primarily over other libs")
	time.sleep(10)
	webbrowser.open("https://github.com/EmbraceLife/LIE/issues/5")

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
def load_plugins(plugin_dir):
	""" Loads the Kur plugins.
	"""
	if plugin_dir:
		if plugin_dir.lower() == 'none':
			return
		Plugin.PLUGIN_DIR = plugin_dir

	for plugin in Plugin.get_enabled_plugins():
		try:
			plugin.load()
		except:
			logger.error('Failed to load plugin "%s". It may not be installed '
				'correctly, or it may contains errors.', plugin)
			raise

###############################################################################
def install_plugin(args):
	""" Installs and enables a plugin.
	"""
	if not Plugin.install(args.plugin):
		return 1

	plugin = Plugin(args.plugin)
	plugin.enabled = True

###############################################################################
def enable_plugin(args):
	""" Enables a plugin.
	"""
	Plugin(args.plugin).enabled = True

###############################################################################
def disable_plugin(args):
	""" Disables a plugin.
	"""
	Plugin(args.plugin).enabled = False

###############################################################################
def list_plugin(args):						# pylint: disable=unused-argument
	""" Lists all enabled plugins.
	"""
	for plugin in Plugin.get_enabled_plugins():
		print(plugin)

###############################################################################
def build_parser():

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
	parser.add_argument('--why', action='store_true',
		help='Display Reasons for why using kur primarily over other libraries.')
	parser.add_argument('--plugin', help='Plugin directory, or "none" to '
		'disable plugins.')

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
	subparser.add_argument('-plot', '--plot_images', action='store_true', help='plot 9 images from a batch of data provider.')
	subparser.set_defaults(func=prepare_data)

	###########################################################################
	# Plugins

	subparser = subparsers.add_parser('plugin', help='Configures Kur plugins.')
	subsubparsers = subparser.add_subparsers(dest='plugin_cmd',
		help='Plugin command')

	subsubparser = subsubparsers.add_parser('install', help='Install a plugin.')
	subsubparser.add_argument('plugin', help='The plugin to install.')
	subsubparser.set_defaults(func=install_plugin)

	subsubparser = subsubparsers.add_parser('enable', help='Enable a plugin.')
	subsubparser.add_argument('plugin', help='The plugin to enable.')
	subsubparser.set_defaults(func=enable_plugin)

	subsubparser = subsubparsers.add_parser('disable',
		help='Disable a plugin.')
	subsubparser.add_argument('plugin', help='The plugin to disable.')
	subsubparser.set_defaults(func=disable_plugin)

	subsubparser = subsubparsers.add_parser('list',
		help='List enabled plugin.')
	subsubparser.set_defaults(func=list_plugin)

	return parser, subparsers

###############################################################################
def parse_args(parser):
	""" Parses command-line arguments.
	"""
	return parser.parse_args()

###############################################################################
def main():
	""" Entry point for the Kur command-line script.
	"""
	gotcha = False
	plugin_dir = None
	for arg in sys.argv[1:]:
		if gotcha:
			plugin_dir = arg
			break
		elif arg == '--plugin':
			gotcha = True
	load_plugins(plugin_dir)

	parser, _ = build_parser()
	args = parse_args(parser)

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
	elif args.why:
		args.func = why
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

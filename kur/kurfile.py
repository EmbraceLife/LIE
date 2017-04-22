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

import sys
import os
import copy
import socket
import warnings
import logging
import glob
from collections import deque

from .engine import ScopeStack, PassthroughEngine
from .reader import Reader
from .containers import Container, ParsingError
from .model import Model, Executor, EvaluationHook, OutputHook, TrainingHook
from .backend import Backend
from .optimizer import Optimizer, Adam
from .loss import Loss
from .providers import Provider, BatchProvider, ShuffleProvider
from .supplier import Supplier
from .utils import mergetools, get_any_value, get_any_key
from .loggers import Logger

logger = logging.getLogger(__name__)

# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule

###############################################################################
class Kurfile:
	""" Class for loading and parsing Kurfiles.
	"""

	DEFAULT_OPTIMIZER = Adam
	DEFAULT_PROVIDER = BatchProvider

	###########################################################################
	def __init__(self, source, engine=None):
		""" Creates a new Kurfile.

			# Arguments

			source: str or dict. If it is a string, it is interpretted as a
				filename to an on-disk Kurfile. Otherwise, it is
				interpretted as an already-loaded, but unparsed, Kurfile.
			engine: Engine instance. The templating engine to use in parsing.
				If None, a Passthrough engine is instantiated.
		"""
		logger.critical("(self, source, engine=None): \n\nCreates a new Kurfile with source and engine: spec \n\n1. get engine assigned JinjaEngine or PassthroughEngine(); \n2. get filenames stored in spec.filename; \n3. Extract everything, all sections into spec.data = parse_source(filename, engine); \n4. create empty storages in spec.containers, spec.model, spec.backend, spec.tempaltes, spec.engine(filled) \n\n Inputs: \n1. source: %s; \n2. engine: %s; \n\n ", source, engine)

		engine = engine or PassthroughEngine()
		if isinstance(source, str):
			filename = os.path.expanduser(os.path.expandvars(source))
			if not os.path.isfile(filename):
				raise IOError('No such file found: {}. Path was expanded to: '
					'{}.'.format(source, filename))
			self.filename = filename
			self.data = self.parse_source(
				engine,
				source=filename,
				context=None
			)
		else:
			self.filename = None
			self.data = dict(source)

		self.containers = None
		self.model = None
		self.backend = None
		self.engine = engine
		self.templates = None

		# logger.debug("(self, source, engine=None): end \n Creates a new Kurfile with source and engine. \n1.get engine assigned JinjaEngine or PassthroughEngine(); \n2. get filenames stored in spec.filename; \n3. parse_source using filename and engine to provide details stored in spec.data; \n4. create empty storages in spec.containers, spec.model, spec.backend, spec.tempaltes, but store engine object inside spec.engine \n\nInputs: \n1. source: %s; \n2. engine: %s; \n\nReturn: what initialized spec.__dict__ contains \n\n ", source, engine)
		# pprint(self.__dict__)
		# print("\n \n")

	###########################################################################
	def parse(self):
		""" Parses the Kurfile.
		"""

		logger.critical("(self): \n\nafter initialize Kurfile object, run spec.parse(): \n\n1. evaluate all section dicts in spec.data with scopes, and reassign them back to spec.data; \n2. section name aliases (like training, testing) are accepted (but I disallowed in my own version); \n3. Extract spec.data['templates'] to spec.templates; \n5. Convert dict spec.data['model'] into Container objects (each layer is a container), create spec.containers = [all the containers] ; \n\nTherefore, spec.data, spec.templates, spec.containers are renewed or filled. \n\n ")

		logger.debug('Parsing Kurfile...')

		# These are reserved section names.
		# The first name in each tuple is the one that we should rely on
		# internally when we reference `self.data` in Kurfile.
		builtin = {
			'settings' : ('settings', ),
			'train' : ('train', ),# 'training'),
			'validate' : ('validate', ),#'validation'),
			'test' : ('test', ),#'testing'),
			'evaluate' : ('evaluate', ),#'evaluation'),
			'templates' : ('templates', ),
			'model' : ('model', ),
			'loss' : ('loss', )
		}

		# The scope stack.
		stack = []

		# Add default entries.
		stack.append({
			'filename' : self.filename,
			'raw' : copy.deepcopy(self.data),
			'parsed' : self.data,
			'host' : socket.gethostname()
		})

		# Parse the settings (backends, globals, hyperparameters, ...)
		self._parse_section(
			self.engine, builtin['settings'], stack, include_key=False,
			auto_scope=True)

		# Parse all of the "action" sections.
		self._parse_section(
			self.engine, builtin['train'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['validate'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['test'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['evaluate'], stack, include_key=True)

		# Parse the templates
		self.templates = self._parse_templates(
			self.engine, builtin['templates'], stack)

		# Parse the model.
		self.containers = self._parse_model(
			self.engine, builtin['model'], stack, required=True)

		# Parse the loss function.
		self._parse_section(
			self.engine, builtin['loss'], stack, include_key=True)

		# Check for unused keys.
		for key in self.data.keys():
			for section in builtin.values():
				if key in section:
					break
			else:
				warnings.warn('Unexpected section in Kurfile: "{}". '
					'This section will be ignored.'.format(key), SyntaxWarning)
		# logger.debug("(self): end \n after initialize Kurfile object, we parse it: \n 1. evaluate all section dicts in spec.data with scopes, and reassign them back to spec.data; \n 2. as a result, spec.data added section aliases (like training, testing); \n 3. other uses here to be answered ....; \n 4. assign spec.data['templates'] to spec.templates; \n 5. convert spec.data['model'] into model as containers, and assign the list of containers inside spec.contaienrs ; \n 6. finally spec's properites are renewed or parsed, done. \n \n Return: what is stored inside spec.__dict__")
		# for k, v in self.__dict__.items():
		# 	print(k, ": \n\t", v)
		# 	print("\n")

	###########################################################################
	def get_model(self, provider=None):
		""" Returns the parsed Model instance.
		"""
		if provider is not None:
			logger.warning("(self, provider=None): \n\nCreate a model object using spec.backend and spec.contaienrs, then Parse the model, and finally Build the model \n\n1. current spec.contaienrs is ready \n2. fill spec.backend with spec.get_backend() \n3. From None, create a model class from spec.containers and spec.backend using Model(backend, containers); \n\n unlike a container, a model's layers are connected backend, and it has tracked input and output connections. But it is just empty at the moment \n\n4. spec.model.parse() set spec.model._parsed True, let engine = PassthroughEngine() if not available, run spec.model.root.parse(engine); \n5. set spec.model.provider = provider; \n6. build(): to fill real objects or values in spec.model.inputs, spec.model.input_aliases, spec.model.outputs, spec.model.output_aliases, spec.model.network; \n\n model is tranformed from dict to a backend-specific layers network  \n\n")

		if self.model is None:
			if self.containers is None:
				raise ValueError('No such model available.')


			# create a kur.Model object, the core is a big dict which store backend object, container objects (all info on each layers), provider, ....
			self.model = Model(
				# get any backend available if not specified
				backend=self.get_backend(),
				# kur's containers for layers are just dict of dict for information on layers
				containers=self.containers
			)

			logger.critical("\n\nspec.model = Model(backend, containers) contains, before parse(), spec.model as follows: \n")
			print("The type of spec.model: \n{}\n\n".format(type(self.model)))
			pprint(self.model.__dict__)
			print("\n\npprint(spec.model.root.__dict__): \n")
			pprint(self.model.root.__dict__)
			print("\n\npprint(spec.model.backend.__dict__): \n")
			pprint(self.model.backend.__dict__)
			print("\n\n")

			# parse() seems only set self.model._parsed to True
			self.model.parse(self.engine)
			# set a specific batch provider
			self.model.register_provider(provider)

			print("After self.model.parse(self.engine), \nand self.model.register_provider(provider), spec.model: \n")
			pprint(self.model.__dict__)
			print("\n\nCheck above, spec.model._parse is set True, spec.model.provider is filled with a provider, and spec.model.root.__dict__['_parsed'] is set True, see spec.model.root below: \n")
			pprint(self.model.root.__dict__)
			print("\n\n")

			logger.critical("\n\nAfter parse the spec.model, let's build the model \n\n ")
			self.model.build()

			logger.critical("Now, self.model.build() fill more properties of spec.model \n\n")
			pprint(self.model.__dict__)
			print("\n\n")

		return self.model

	###########################################################################
	def get_backend(self):
		""" Creates a new backend from the Kurfile.

			# Return value

			Backend instance

			# Notes

			- If no "global" section exists, or if no "backend" section exists
			  in the "global" section, this will try to instantiate any
			  supported backend installed on the system.
			- This passes `self.data['global']['backend']` to the Backend
			  factory method, if such keys exist; otherwise, it passes None to
			  the factory method.
		"""
		if self.backend is None:
			self.backend = Backend.from_specification(
				(self.data.get('settings') or {}).get('backend')
			)
		return self.backend

	###########################################################################
	def get_seed(self):
		""" Retrieves the global random seed, if any.

			# Return value

			The `settings.seed` value, as an integer, if such a key exists.
			If the key does not exist, returns None.
		"""

		if 'settings' not in self.data or \
				not isinstance(self.data['settings'], dict):
			return None

		seed = self.data['settings'].get('seed')
		if seed is None:
			return None
		elif not isinstance(seed, int):
			raise ValueError('Unknown format for random seed: {}'.format(seed))

		if seed < 0 or seed >= 2**32:
			logger.warning('Random seeds should be unsigned, 32-bit integers. '
				'We are truncating the seed.')
			seed %= 2**32

		return seed

	###########################################################################
	def get_provider(self, section, accept_many=False):
		""" Creates the provider corresponding to a part of the Kurfile.

			# Arguments

			section: str. The name of the section to load the provider for.
			accept_many: bool (default: False). If True, more than one data
				supplier is allowed to be specified; otherwise, an exception is
				raised if the data supplier is missing.

			# Return value

			If the section exists and has at least the "data" or "provider"
			section defined, then this returns a Provider instance. Otherwise,
			returns None.
		"""
		logger.info("(self, section, accept_many=False):  \n\nUsing detailed info from spec.data[section]['data'] to build data suppliers first, then buid a data provider : \n\n1. get spec.data[section]; \n2. make sure it has a key as 'data' or 'provider'; \n3. store spec.data[section]['data'] in 'supplier_list'; \n4. consider when there are more than one data sources; \n5. create data Supplier object from spec.data[section]['data']; \n6. create data provider using data supplier and provider_detailed_info from spec.data[section]['provider']; \n7. finally return this provider \n\nInputs: \n1. section: %s \n2. accept_many: %s \n\n", section, accept_many)

		if section in self.data:
			section = self.data[section]
			if not any(k in section for k in ('data', 'provider')):
				return None
		else:
			return None

		supplier_list = section.get('data') or {}
		if isinstance(supplier_list, (list, tuple)):
			supplier_list = {'default' : supplier_list}
		elif not isinstance(supplier_list, dict):
			raise ValueError('"data" section should be a list or dictionary.')

		if not accept_many and len(supplier_list) > 1:
			raise ValueError('We only accept a single "data" entry for this '
				'section, but found {}.'.format(len(supplier_list)))

		logger.critical("\n\nspec.data[section]['data'] for building data suppliers are: \n")
		pprint(supplier_list)
		print("\n\n")

		suppliers = {}
		for k, v in supplier_list.items():
			if not isinstance(v, (list, tuple)):
				raise ValueError('Data suppliers must form a list of '
					'suppliers.')
			suppliers[k] = [
				Supplier.from_specification(entry, kurfile=self)
				for entry in v
			]


		provider_spec = dict(section.get('provider') or {})

		logger.critical("\n\nspec.data[section]['provider'] for building data provider are: \n")
		pprint(provider_spec)
		print("\n\n")

		if 'name' in provider_spec:
			provider = Provider.get_provider_by_name(provider_spec.pop('name'))
		else:
			provider = Kurfile.DEFAULT_PROVIDER

		final_providers = {
			k: provider(
			    sources=Supplier.merge_suppliers(v),
			    **provider_spec
			)
			for k, v in suppliers.items()
		}


		return final_providers

	###########################################################################
	def get_training_function(self):
		""" Returns a function that will train the model.
		"""
		logger.debug("(self): \nreturn the training function: \n1. make sure the spec.data has a 'train' section; \n2. If log exist in train section, create a new hook based on it; \n3. get number of epochs assigned, get stop_when assigned or as {}, when both epochs and stop_when exist, stop_when has priority. stop_when has epoch_number and mode; \n4. get a provider out of providers; \n5. get training_hooks dict from train section (a list) and create training_hooks objects; \n6. If validate section is available, get all data providers in validate section, and get validate_weights, and set validat_weights to be best_valid weights; \n7. If validate section has hooks, get all hook objects into a list; \n8. get train_weights, if it is string, set train_weights to be initial weights; if it is a dict, train_weights has 3 elements: initial, best, and last weights; \n9. get file path for initial_weights, best_train weights, best_valid and last_weights;... continue ...")

		logger.warning("\n10. build model object using data provider to spec, then build Executor trainer; \n11. create the actual training function: 1. if initial_weights is available, then restore the weights to model; 2. store all weights paths, data provider, hooks, checkpoint in a dict named defaults; 3. update this list with input arguments; 4. train the Executor trainer with defaults dict; \n12. return this actual training function.\n\n")


		if 'train' not in self.data:
			raise ValueError('Cannot construct training function. There is a '
				'missing "train" section.')

		if 'log' in self.data['train']:
			log = Logger.from_specification(self.data['train']['log'])
		else:
			log = None

		epochs = self.data['train'].get('epochs')
		stop_when = self.data['train'].get('stop_when', {})
		if epochs:
			if stop_when:
				warnings.warn('"stop_when" has replaced "epochs" in the '
					'"train" section. We will try to merge things together, '
					'giving "stop_when" priority.', DeprecationWarning)
			if isinstance(epochs, dict):
				if 'number' in epochs and 'epochs' not in stop_when:
					stop_when['epochs'] = epochs['number']
				if 'mode' in epochs and 'mode' not in stop_when:
					stop_when['mode'] = epochs['mode']
			elif 'epochs' not in stop_when:
				stop_when['epochs'] = epochs

		provider = get_any_value(self.get_provider('train'))

		training_hooks = self.data['train'].get('hooks') or []
		if not isinstance(training_hooks, (list, tuple)):
			raise ValueError('"hooks" (in the "train" section) should '
				'be a list of hook specifications.')
		training_hooks = [TrainingHook.from_specification(spec) \
			for spec in training_hooks]

		if 'validate' in self.data:
			validation = self.get_provider('validate', accept_many=True)
			validation_weights = self.data['validate'].get('weights')
			if validation_weights is None:
				best_valid = None
			elif isinstance(validation_weights, str):
				best_valid = validation_weights
			elif isinstance(validation_weights, dict):
				best_valid = validation_weights.get('best')
			else:
				raise ValueError('Unknown type for validation weights: {}'
					.format(validation_weights))

			validation_hooks = self.data['validate'].get('hooks', [])
			if not isinstance(validation_hooks, (list, tuple)):
				raise ValueError('"hooks" (in the "validate" section) should '
					'be a list of hook specifications.')
			validation_hooks = [EvaluationHook.from_specification(spec) \
				for spec in validation_hooks]
		else:
			validation = None
			best_valid = None
			validation_hooks = None

		train_weights = self.data['train'].get('weights')
		if train_weights is None:
			initial_weights = best_train = last_weights = None
			deprecated_checkpoint = None
		elif isinstance(train_weights, str):
			initial_weights = train_weights
			best_train = train_weights if best_valid is None else None
			last_weights = None
			initial_must_exist = False
			deprecated_checkpoint = None
		elif isinstance(train_weights, dict):
			initial_weights = train_weights.get('initial')
			best_train = train_weights.get('best')
			last_weights = train_weights.get('last')
			initial_must_exist = train_weights.get('must_exist', False)
			deprecated_checkpoint = train_weights.get('checkpoint')
		else:
			raise ValueError('Unknown weight specification for training: {}'
				.format(train_weights))

		checkpoint = self.data['train'].get('checkpoint')

		if deprecated_checkpoint is not None:
			warnings.warn('"checkpoint" belongs under "train", not under '
				'"weights".', DeprecationWarning)
			if checkpoint is None:
				checkpoint = deprecated_checkpoint
			else:
				logger.warning('The currently-accepted "checkpoint" will be '
					'used over the deprecated "checkpoint".')

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		initial_weights, best_train, best_valid, last_weights = [
			expand(x) if x is not None else x for x in
				(initial_weights, best_train, best_valid, last_weights)
		]

		model = self.get_model(provider)
		trainer = self.get_trainer()

		def func(**kwargs):
			""" Trains a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				elif initial_must_exist:
					logger.error('Configuration indicates that the weight '
						'file must exist, but the weight file was not found.')
					raise ValueError('Missing initial weight file. If you '
						'want to proceed anyway, set "must_exist" to "no" '
						'under the training "weights" section.')
				else:
					if log is not None and \
						(log.get_number_of_epochs() or 0) > 0:
						logger.warning('The initial weights are missing: %s. '
							'However, the logs suggest that this model has '
							'been trained previously. It is possible that the '
							'log file is corrupt/out-of-date, or that the '
							'weight files are not supposed to be missing.',
							initial_weights)
					else:
						logger.debug('Ignoring missing initial weights: %s. If '
							'this is undesireable, set "must_exist" to "yes" '
							'in the approriate "weights" section.',
							initial_weights)
			defaults = {
				'provider' : provider,
				'validation' : validation,
				'stop_when' : stop_when,
				'log' : log,
				'best_train' : best_train,
				'best_valid' : best_valid,
				'last_weights' : last_weights,
				'training_hooks' : training_hooks,
				'validation_hooks' : validation_hooks,
				'checkpoint' : checkpoint
			}
			defaults.update(kwargs)
			return trainer.train(**defaults)

		return func

	###########################################################################
	@staticmethod
	def find_default_provider(providers):
		""" Finds the provider that will be used for constructing the model.
		"""
		if 'default' in providers:
			return providers['default']
		k = get_any_key(providers)
		logger.debug('Using multiple providers. Since there is no '
			'"default" provider, the default one we will use to construct '
			'the model is: %s', k)
		return providers[k]

	###########################################################################
	def get_testing_function(self):
		""" Returns a function that will test the model.
		"""
		if 'test' not in self.data:
			raise ValueError('Cannot construct testing function. There is a '
				'missing "test" section.')

		providers = self.get_provider('test', accept_many=True)
		default_provider = self.find_default_provider(providers)

		# No reason to shuffle things for this.
		for provider in providers.values():
			if isinstance(provider, ShuffleProvider):
				provider.randomize = False

		weights = self.data['test'].get('weights')
		if weights is None:
			initial_weights = None
		elif isinstance(weights, str):
			initial_weights = weights
		elif isinstance(weights, dict):
			initial_weights = weights.get('initial')
		else:
			raise ValueError('Unknown weight specification for testing: {}'
				.format(weights))

		hooks = self.data['test'].get('hooks', [])
		if not isinstance(hooks, (list, tuple)):
			raise ValueError('"hooks" (in the "test" section) should be a '
			   'list of hook specifications.')
		hooks = [EvaluationHook.from_specification(spec) for spec in hooks]

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		if initial_weights is not None:
			initial_weights = expand(initial_weights)

		model = self.get_model(default_provider)
		trainer = self.get_trainer(with_optimizer=False)

		def func(**kwargs):
			""" Tests a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				else:
					logger.error('No weight file found: %s. We will just '
						'proceed with default-initialized weights. This will '
						'test that the system works, but the results will be '
						'terrible.', initial_weights)
			else:
				logger.warning('No weight file specified. We will just '
					'proceed with default-initialized weights. This will '
					'test that the system works, but the results will be '
					'terrible.')
			defaults = {
				'providers' : providers,
				'validating' : False,
				'hooks' : hooks
			}
			defaults.update(kwargs)
			return trainer.test(**defaults)

		return func

	###########################################################################
	def get_trainer(self, with_optimizer=True):
		""" Creates a new Trainer from the Kurfile.

			# Return value

			Trainer instance.
		"""
		return Executor(
			model=self.get_model(),
			loss=self.get_loss(),
			optimizer=self.get_optimizer() if with_optimizer else None
		)

	###########################################################################
	def get_optimizer(self):
		""" Creates a new Optimizer from the Kurfile.

			# Return value

			Optimizer instance.
		"""
		if 'train' not in self.data:
			raise ValueError('Cannot construct optimizer. There is a missing '
				'"train" section.')

		spec = self.data['train'].get('optimizer', {})
		if isinstance(spec, str):
			spec = {'name' : spec}
		elif not isinstance(spec, dict):
			raise ValueError('Invalid value for optimizer: {}'.format(spec))
		else:
			spec = dict(spec)

		if 'name' in spec:
			optimizer = Optimizer.get_optimizer_by_name(spec.pop('name'))
		else:
			optimizer = Kurfile.DEFAULT_OPTIMIZER

		return optimizer(**spec)

	###########################################################################
	def get_loss(self):
		""" Creates a new Loss from the Kurfile.

			# Return value

			If 'loss' is defined, then this returns a dictionary whose keys are
			output layer names and whose respective values are Loss instances.
			Otherwise, this returns None.
		"""
		if 'loss' not in self.data:
			return None

		spec = self.data['loss']
		if not isinstance(spec, (list, tuple)):
			raise ValueError('"loss" section expects a list of loss '
				'functions.')

		result = {}
		for entry in spec:
			entry = dict(entry)
			target = entry.pop('target', None)
			if target is None:
				raise ValueError('Each loss function in the "loss" list '
					'must have a "target" entry which names the output layer '
					'it is attached to.')

			name = entry.pop('name', None)
			if name is None:
				raise ValueError('Each loss function in the "loss" list '
					'must have a "name" entry which names the loss function '
					'to use for that output.')

			if target in result:
				logger.warning('A loss function for the "%s" output was '
					'already defined. We will replace it by this new one we '
					'just found.', target)
			result[target] = Loss.get_loss_by_name(name)(**entry)

		return result

	###########################################################################
	def get_evaluation_function(self):
		""" Returns a function that will evaluate the model.
		"""
		if 'evaluate' not in self.data:
			raise ValueError('Cannot construct evaluation function. There is '
				'a missing "evaluate" section.')

		provider = get_any_value(self.get_provider('evaluate'))

		# No reason to shuffle things for this.
		if isinstance(provider, ShuffleProvider):
			provider.randomize = False

		weights = self.data['evaluate'].get('weights')
		if weights is None:
			initial_weights = None
		elif isinstance(weights, str):
			initial_weights = weights
		elif isinstance(weights, dict):
			initial_weights = weights.get('initial')
		else:
			raise ValueError('Unknown weight specification for evaluation: {}'
				.format(weights))

		destination = self.data['evaluate'].get('destination')
		if isinstance(destination, str):
			destination = OutputHook(path=destination)
		elif isinstance(destination, dict):
			destination = OutputHook(**destination)
		elif destination is not None:
			ValueError('Expected a string or dictionary value for '
				'"destination" in "evaluate". Received: {}'.format(
					destination))

		hooks = self.data['evaluate'].get('hooks', [])
		if not isinstance(hooks, (list, tuple)):
			raise ValueError('"hooks" should be a list of hook '
				'specifications.')
		hooks = [EvaluationHook.from_specification(spec) for spec in hooks]

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		if initial_weights is not None:
			initial_weights = expand(initial_weights)

		model = self.get_model(provider)
		evaluator = self.get_evaluator()

		def func(**kwargs):
			""" Evaluates a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				else:
					logger.error('No weight file found: %s. We will just '
						'proceed with default-initialized weights. This will '
						'test that the system works, but the results will be '
						'terrible.', initial_weights)
			else:
				logger.warning('No weight file specified. We will just '
					'proceed with default-initialized weights. This will '
					'test that the system works, but the results will be '
					'terrible.')
			defaults = {
				'provider' : provider,
				'callback' : None
			}
			defaults.update(kwargs)
			result, truth = evaluator.evaluate(**defaults)
			orig = (result, truth)
			prev = orig
			for hook in hooks:
				new_prev = hook.apply(prev, orig, model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev
			if destination is not None:
				new_prev = destination.apply(prev, orig, model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev
			return prev

		return func

	###########################################################################
	def get_evaluator(self):
		""" Creates a new Evaluator from the Kurfile.

			# Return value

			Evaluator instance.
		"""
		return Executor(
			model=self.get_model()
		)

	###########################################################################
	def _parse_model(self, engine, section, stack, *, required=True):
		""" Parses the top-level "model" entry.

			# Arguments

			engine: Engine instance. The templating engine to use in
				evaluation.
			section: str or list of str. The key indicating which top-level
				entry to parse. If a list, the first matching key is used; this
				allows for multiple aliases to the same section.
			stack: list. The list of scope dictionaries to use in evaluation.
			required: bool (default: True). If True, an exception is raised if
				the key is not present in the Kurfile.

			# Return value

			If the model section is found, the list of containers is returned.
			If the model section is not found but is required, an exception is
			raised. If the model section is not found and isn't required, None
			is returned.
		"""
		if isinstance(section, str):
			section = (section, )

		key = None		# Not required, but shuts pylint up.
		for key in section:
			if key in self.data:
				break
		else:
			if required:
				raise ValueError('Missing required section: {}'
					.format(', '.join(section)))
			else:
				return None

		if not isinstance(self.data[key], (list, tuple)):
			raise ValueError(
				'Section "{}" should contain a list of layers.'.format(key))

		logger.debug("(engine, section, stack, required=True): start \nConvert model dicts from spec.data['model'] to a list of containers of all layers: \n1. stack: provide scopes for evaulation; \n2. section: a string for top level sections, here to be 'model'; \n3. make sure 'model' is a key found in spec.data; \n4. for each dict element or entry of spec.data['model'], do JinjaEngine.evaluate, then create a container object; \n5. parse container with engine, and store all containers in a list; \n6. return the list of containers. \n\n Parse model from spec.data['model']: \n\n Inputs: \n1. engine: %s; \n2. section: %s; \n3. stack: %s; \n4. required: (default True) %s \n\n Return: a list of container objects \n ", engine, section, [item.keys() for item in stack], required)
		# print("As part of Inputs: spec.data['model'] is a dict: see below")
		# pprint(self.data[key])
		# print("\n\n")

		with ScopeStack(engine, stack):
			queue = deque(self.data[key])
			containers = []
			while queue:
				entry = queue.popleft()
				entry = engine.evaluate(entry)
				containers.append(
					Container.create_container_from_data(entry)
				)
			for container in containers:
				container.parse(engine)

		logger.debug("(engine, section, stack, required=True): end \n Convert model dicts from spec.data['model'] to a list of containers of all layers: \n 1. stack: provide scopes for evaulation; \n 2. section: a string for top level sections, here to be 'model'; \n 3. make sure 'model' is a key found in spec.data; \n 4. for each dict element or entry of spec.data['model'], do JinjaEngine.evaluate, then create a container object; \n 5. parse container with engine, and store all containers in a list; \n 6. return the list of containers. \n \n Parse model from spec.data['model']: \n \n Inputs: \n 1. engine: %s; \n 2. section: %s; \n 3. stack: %s; \n 4. required: (default True) %s \n \n Return: a list of container objects \n %s \n \n ", engine, section, [item.keys() for item in stack], required, containers)

		return containers

	###########################################################################
	def parse_source(self, engine, *,
		source=None, context=None, loaded=None):
		""" Parses a source, and its includes (recursively), and returns the
			merged sources.
		"""
		logger.critical("(engine, source=filename, context=None): \n\nExtract kurfile.yml content into a large dict for spec.data : \n\n1. read dictionary from a yml file, and its-includes files (recursively); \n2. merge the dictionaries together; \n3. returns the merged data. \n  \n Input args: \n\n1. engine: %s \n2. source: %s \n \n ", engine, source)

		def get_id(filename):
			""" Returns a tuple which uniquely identifies a file.

				This function follows symlinks.
			"""
			filename = os.path.expanduser(os.path.expandvars(filename))
			stat = os.stat(filename)
			return (stat.st_dev, stat.st_ino)

		# Process the "loaded" iterable.
		loaded = loaded or set()

		if isinstance(source, str):
			filename = source
			strategy = None
		elif isinstance(source, dict):
			if 'source' not in source:
				raise ParsingError('Error while parsing source file {}. '
					'Missing required "source" key in the include '
					'dictionary: {}.'.format(context or 'top-level', source))
			filename = source.pop('source')
			strategy = source.pop('method', None)
			for k in source:
				warnings.warn('Warning while parsing source file {}. '
					'Ignoring extra key in an "include" dictionary: {}.'
					.format(context or 'top-level', k))
		else:
			raise ParsingError('Error while parsing source file {}. '
				'Expected each "include" to be a string or a dictionary. '
				'Received: {}'.format(context or 'top-level', source))

		logger.debug('Parsing source: %s, included by %s.', filename,
			context or 'top-level')

		if context:
			filename = os.path.join(os.path.dirname(context), filename)
		expanded = os.path.expanduser(os.path.expandvars(filename))
		if not os.path.isfile(expanded):
			raise IOError('Error while parsing source file {}. No such '
				'source file found: {}. Path was expanded to: {}'.format(
				context or 'top-level', filename, expanded))
		file_id = get_id(expanded)
		if file_id in loaded:
			logger.warning('Skipping an already included source file: %s. '
				'This may have unintended consequences, since the merge '
				'order may result in different final results. You should '
				'refactor your Kurfile to avoid circular includes.',
				expanded)
			return {}
		else:
			loaded.add(file_id)

		try:
			data = Reader.read_file(expanded)
		except:
			logger.exception('Failed to read file: %s. Check your syntax.',
				expanded)
			raise

		new_sources = data.pop('include', [])
		engine.evaluate(new_sources, recursive=True)
		if isinstance(new_sources, str):
			new_sources = [new_sources]
		elif not isinstance(new_sources, (list, tuple)):
			raise ValueError('Error while parsing file: {}. All "include" '
				'sections must be a single source file or a list of source '
				'files. Instead we received: {}'.format(
					expanded, new_sources))

		def load_source(source):
			return mergetools.deep_merge(
				self.parse_source(
					engine,
					source=source,
					context=expanded,
					loaded=loaded
				),
				data,
				strategy=strategy
			)
		for new_source in new_sources:
			if isinstance(new_source, dict) and 'source' in new_source:
				sub_sources = new_source.pop('source')
				if not isinstance(sub_sources, (list, tuple)):
					sub_sources = [sub_sources]
				for sub_source in sub_sources:
					for x in self.glob(sub_source, source=expanded,
						recursive=True
					):
						new_source['source'] = x
						data = load_source(dict(new_source))
			elif isinstance(new_source, str):
				for x in self.glob(new_source, source=expanded,
					recursive=True
				):
					data = load_source(x)
			else:
				data = load_source(new_source)

		# logger.debug("(engine, source=filename, context=None): end \n Extract kurfile.yml content into a large dict for spec.data : \n 1. read dictionary from a yml file, and its-includes files (recursively); \n 2. merge the dictionaries together; \n 3. returns the merged data. \n  \n Input args: \n \t 1. engine: %s \n \t 2. source: %s \n \n Return: \n \t data as %s, with following keys: %s \n \t The content: \n  ", engine, source, type(data), data.keys())
		# for k, v in data.items():
		# 	print(k, ": \n \t", v)
		# print("\n \n")

		return data

	###########################################################################
	@staticmethod
	def glob(path, *, source=None, recursive=False):
		""" Wrapper for Python's `glob`.
		"""
		if sys.version_info < (3, 5):
			kwargs = {}
			if recursive and '**' in path:
				warnings.warn('Recursive globbing is not supported on Python '
					'3.4. Ignoring this...', UserWarning)
		else:
			kwargs = {'recursive' : recursive}

		if source:
			path = os.path.join(os.path.dirname(source), path)
		yield from glob.iglob(path, **kwargs)

	###########################################################################
	def _parse_templates(self, engine, section, stack):
		""" Parses the template section.
		"""
		logger.debug("(self, engine, section, stack): start \n return the templates dict from spec.data['templates']: \n 1. section should be a string as 'templates'; \n 2. 'templates' should be a key in spec.data; \n 3. spec.data['templates'] should be a dict; \n 4. keys inside spec.data['templates'] should not be a real container object name or real layer object name; \n 5. engine object has _scope, _templates, env, state; only env is not Empty; fill _templates with spec.data['templates']; \n 6. return spec.data['templates'] \n \n Parse template: %s \n\n\n Inputs: \n1. engine: %s; \n2. section: %s; \n3. stack: %s \n \n", section[0], engine, section, [item.keys() for item in stack])

		if isinstance(section, str):
			section = (section, )

		key = None		# Not required, but shuts pylint up.
		for key in section:
			if key in self.data:
				break
		else:
			return None

		if not self.data[key]:
			return None

		if not isinstance(self.data[key], dict):
			raise ValueError('Section "{}" should contain a dictionary of '
				'templates.'.format(key))

		for template in self.data[key]:
			other = Container.get_container_for_name(template)
			if other:
				raise ValueError('Templates cannot override built-in layers. '
					'We found a conflict with template "{}".'.format(template))

		engine.register_templates(self.data[key])

		logger.debug("(self, engine, section, stack): end \nreturn the templates dict from spec.data['templates']: \n1. section should be a string as 'templates'; \n2. 'templates' should be a key in spec.data; \n3. spec.data['templates'] should be a dict; \n4. keys out of spec.data['templates'] should not be a real container object name or real layer object name; \n5. engine object has _scope, _templates, env, state; only env is not Empty; fill _templates with spec.data['templates']; \n6. return spec.data['templates'] \n \n Parse template: %s \n\n Inputs: \n1. engine: %s; \n2. section: %s; \n3. stack is a huge dict: %s \n \n", section[0], engine, section, [item.keys() for item in stack])
		# print("engine has _templates filled: ")
		# pprint(engine.__dict__)
		# print("\n \n")

		return self.data[key]

	###########################################################################
	def _parse_section(self, engine, section, stack, *,
		required=False, include_key=True, auto_scope=False):
		""" Parses a single top-level entry in the Kurfile.

			# Arguments

			engine: Engine instance. The templating engine to use in
				evaluation.
			section: str or list of str. The key indicating which top-level
				entry to parse. If a list, the first matching key is used; this
				allows for multiple aliases to the same section.
			stack: list. The list of scope dictionaries to use in evaluation.
			required: bool (default: False). If True, an exception is raised if
				the key is not present in the Kurfile.
			include_key: bool (default: False). If True, the parsed section is
				added to the scope stack as a dictionary with a single item,
				whose key is the name of the parsed section, and whose value is
				the evaluated section. If False, then the evaluated section is
				added directly to the scope stack (it must evaluate to a
				dictionary for this to work).

			# Return value

			The parsed section.

			# Note

			- This will replace the values of `self.data` with the evaluated
			  version.
		"""
		if isinstance(section, str):
			section = (section, )

		logger.debug('(self, engine, section, stack, *, required=False, include_key=True, auto_scope=False): \nEvaluate a Kurfile section as assign it back to spec.data[section]: \n \t parse section: %s \n ', section[0])

		key = None		# Not required, but shuts pylint up.
		for key in section:
			if key in self.data:
				break
		else:
			if required:
				raise ValueError(
					'Missing required section: {}'.format(', '.join(section)))
			else:
				return None

		extra_stack = [{key : self.data[key]}]
		if auto_scope:
			extra_stack += [self.data[key]] \
				if isinstance(self.data[key], dict) else []
		with ScopeStack(engine, stack + extra_stack):
			evaluated = engine.evaluate(self.data[key], recursive=True)

		if not include_key:
			if evaluated is not None:
				if not isinstance(evaluated, dict):
					raise ValueError(
						'Section "{}" should contain key/value pairs.'
						.format(key))
				stack.append(evaluated)
		stack.append({key : evaluated})

		# Now, just in case this was aliased, let's make sure we keep our
		# naming scheme consistent.
		for key in section:
			self.data[key] = evaluated

		return evaluated

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

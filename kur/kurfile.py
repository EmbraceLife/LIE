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
        """ (source_str_or_dict_as_loaded_kurfile, engine=None): Creates a new Kurfile with source and engine. 1.return: Kurfile object with a bunch of properties: filename, data, containers, model, backend, engine, templates; Note: only filename, engine, data are not None right now

                # Arguments

                source: str or dict. If it is a string, it is interpretted as a
                        filename to an on-disk Kurfile. Otherwise, it is
                        interpretted as an already-loaded, but unparsed, Kurfile.
                engine: Engine instance. The templating engine to use in parsing.
                        If None, a Passthrough engine is instantiated.
        """

        # logger.warning("(source_str_or_dict_as_loaded_kurfile, engine=None): Creates a new Kurfile with source and engine. 1.return: Kurfile object with a bunch of properties: filename, data, containers, model, backend, engine, templates; Note: only filename, engine, data are not None right now");

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

    ###########################################################################
    def parse(self):
        """ (self): after initialize Kurfile object, we parse it: 1. evaluate all section dicts in spec.data with scopes; 2. as a result, spec.data added section aliases (like training, testing);3. other uses here to be answered ....; 4. assign spec.data['templates'] to spec.templates; 3. convert spec.data['model'] into model as containers, and store the list of containers inside spec.contaienrs ; 5. return Nothing
        """

        # logger.warning("(self): after initialize Kurfile object, we parse it: 1. evaluate all section dicts in spec.data with scopes; 2. as a result, spec.data added section aliases (like training, testing);3. other uses here to be answered ....; 4. assign spec.data['templates'] to spec.templates; 3. convert spec.data['model'] into model as containers, and store the list of containers inside spec.contaienrs ; 5. return Nothing")

        # These are reserved section names stored in spec.data
        # The first name in each tuple is the one that we should rely on
        # internally when we reference `self.data` in Kurfile.
        builtin = {
            'settings': ('settings', ),
            'train': ('train', ), #'training'),
            'validate': ('validate', ), #'validation'),
            'test': ('test', ), #'testing'),
            'evaluate': ('evaluate', ), #'evaluation'),
            'templates': ('templates', ),
            'model': ('model', ),
            'loss': ('loss', )
        }

        # The scope stack.
        stack = []

        # Add default entries.
        stack.append({
            'filename': self.filename,
            'raw': copy.deepcopy(self.data),
            'parsed': self.data,
            'host': socket.gethostname()
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

        # assign spec.data['templates'] to spec.templates
        self.templates = self._parse_templates(
            self.engine, builtin['templates'], stack)

        # convert model as dict into model as containers, and store the list of containers inside spec.contaienrs
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
                              'This section will be ignored.', SyntaxWarning)

    ###########################################################################
    def get_model(self, provider=None):
        """ Returns the parsed Model instance: 1. a parsed Kurfile object spec has model and backend as None; 2. build a Model object for spec.model with backend and containers; now spec.model and spec.backend are no more None; 3. parse spec.model with engine, set spec.model['_parsed'] to True; 4. register spec.model['provider'] to be BatchProvider; 5. build spec.model by initialize spec.model['input_aliases'] and spec.model['inputs'] with real content
        """

		# after parse() the Kurfile object, spec.data, templates, containers have been processed, leaving spec.model and spec.backend as None
        if self.model is None:
            if self.containers is None:
                raise ValueError('No such model available.')

			# build kur.model.Model object with backend and containers
			# note: many of __dict__ insides are None or {}
            self.model = Model(
                backend=self.get_backend(),
                containers=self.containers
            )
			# parse spec.model:
			# but all it does: seemingly only set a dict spec.model['_parsed'] from false to true
            self.model.parse(self.engine)
			# register_provider for spec.model:
			# now, spec.model['provider'] is set with BatchProvider
            self.model.register_provider(provider)
			# after build model, spec.model['input_aliases'] set {'images:': 'images'}; spec.model['inputs'] set to be OrderedDict
            self.model.build()
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
        """ Get data from spec.data[section]['data'] as a dict all the way to data provider object: 1. get spec.data[section]; 2. make sure it has a key as 'data' or 'provider'; 3. store spec.data[section]['data'] in 'supplier_list'; 4. set `accept_many` true if there are more than 1 data providers; 5. get real data supplier objects using spec.data[section]['data'] the dict (keep 8 data sources, extract 3 into {'data': 3 selected sources); 6. get real data provider instances using spec.data[section]['data'] and spec.data[section]['provider'], and make all data sources ready to use for data provider; 7. finally return this provider

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

        # get the dict under selected section in spect.data,
        # assign the dict to variable `section`
        # make sure such dict has either 'data' or 'provider' as keys
        if section in self.data:
            section = self.data[section]
            if not any(k in section for k in ('data', 'provider')):
                return None
        else:
            return None

        # section['data'] is a list, contain one or more data source info
        # assign {'default': section['data']} to 'supplier_list'
        # section.get('provider') provide some info on how we want a provider
        supplier_list = section.get('data') or {}
        if isinstance(supplier_list, (list, tuple)):
            supplier_list = {'default': supplier_list}
        elif not isinstance(supplier_list, dict):
            raise ValueError('"data" section should be a list or dictionary.')

        # set `accept_many` true if there are more than 1 data providers
        if not accept_many and len(supplier_list) > 1:
            raise ValueError('We only accept a single "data" entry for this '
                             'section, but found {}.'.format(len(supplier_list)))

        # Extract one or more Supplier objects from spec.data[section]['data']
		# in speech.yml case, there is only one SpeechRecognitionSupplier object, but it has 8 data sources, only extracted 3 data_dicts (transcript, audio, duration) with length 2432 into this SpeechRecognitionSupplier_object.data
		# the other 5 data sources are not exacted yet, but it is inside this same supplier object
        suppliers = {}
        for k, v in supplier_list.items():
            if not isinstance(v, (list, tuple)):
                raise ValueError('Data suppliers must form a list of '
                                 'suppliers.')
            suppliers[k] = [
                # extract a SpeechRecognitionSupplier object from a spec.data[section]['data']
                Supplier.from_specification(entry, kurfile=self)
                for entry in v
            ]

        # get additional info about data provider, type as dict
        provider_spec = dict(section.get('provider') or {})

        # if data provider has a 'name' key, then get a provider class according
        # to name, otherwise use BatchProvider class instead
		# BatchProvider has default batch_size as 32
        if 'name' in provider_spec:
            provider = Provider.get_provider_by_name(provider_spec.pop('name'))
        else:
            provider = Kurfile.DEFAULT_PROVIDER

        # In speech case, there is only one SpeechRecognitionSupplier object, if there is more then collect all data sources of one or more supplier objects as `sources` for initializing a provider
		# then initialize a BatchProvider object with the SpeechRecognitionSupplier object and provider_spec details
		# now provider has all 8 data sources to provide for batches

        # p = None
        # for k, v in suppliers.items():
	    #     p = provider(
	    #         sources=Supplier.merge_suppliers(v),
	    #         **provider_spec
	    #     )

        return {
            k: provider(
                sources=Supplier.merge_suppliers(v),
                **provider_spec
            )
            for k, v in suppliers.items()
        }

    ###########################################################################
    def get_training_function(self):
        """ Returns a function that will train the model.
        """
        logger.warning("(self): return the training function: 1. make sure the spec.data has a 'train' section; 2. If log exist in train section, create a new hook based on it; 3. get number of epochs assigned, get stop_when assigned or as {}, when both epochs and stop_when exist, stop_when has priority. stop_when has epoch_number and mode; 4. get a provider out of providers; 5. get training_hooks dict from train section (a list) and create training_hooks objects; 6. If validate section is available, get all data providers in validate section, and get validate_weights, and set validat_weights to be best_valid weights; 7. If validate section has hooks, get all hook objects into a list; 8. get train_weights, if it is string, set train_weights to be initial weights; if it is a dict, train_weights has 3 elements: initial, best, and last weights; 9. get file path for initial_weights, best_train weights, best_valid and last_weights;... continue ...")
        logger.warning("10. build model object using data provider to spec, then build Executor trainer; 11. create the actual training function: 1. if initial_weights is available, then restore the weights to model; 2. store all weights paths, data provider, hooks, checkpoint in a dict named defaults; 3. update this list with input arguments; 4. train the Executor trainer with defaults dict; 12. return this actual training function.")

		# make sure the spec.data has a 'train' section
        if 'train' not in self.data:
            raise ValueError('Cannot construct training function. There is a '
                             'missing "train" section.')

		# If log exist in train section, create a new hook based on it
        if 'log' in self.data['train']:
            log = Logger.from_specification(self.data['train']['log'])
        else:
            log = None

		# get number of epochs assigned
        epochs = self.data['train'].get('epochs')

		# get stop_when assigned or as {}
        stop_when = self.data['train'].get('stop_when', {})

		# when both epochs and stop_when exist, stop_when has priority
        if epochs:
            if stop_when:
                warnings.warn('"stop_when" has replaced "epochs" in the '
                              '"train" section. We will try to merge things together, '
                              'giving "stop_when" priority.', DeprecationWarning)

		    # if epoch is a dict, store epoch_number and epoch_mode inside stop_when dict
            if isinstance(epochs, dict):
                if 'number' in epochs and 'epochs' not in stop_when:
                    stop_when['epochs'] = epochs['number']
                if 'mode' in epochs and 'mode' not in stop_when:
                    stop_when['mode'] = epochs['mode']
			# if epochs is not a dict, nor a key in stop_when, just store epochs in stop_when
            elif 'epochs' not in stop_when:
                stop_when['epochs'] = epochs

		# get a provider out of providers
        provider = get_any_value(self.get_provider('train'))

		# get training_hooks dict from train section (a list) and create training_hooks objects
        training_hooks = self.data['train'].get('hooks') or []
        if not isinstance(training_hooks, (list, tuple)):
            raise ValueError('"hooks" (in the "train" section) should '
                             'be a list of hook specifications.')
        training_hooks = [TrainingHook.from_specification(spec)
                          for spec in training_hooks]

	    # If validate section is available, get all data providers in validate section, and get validate_weights, and set validat_weights to be best_valid weights
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

			# If validate section has hooks, get all hook objects into a list
            validation_hooks = self.data['validate'].get('hooks', [])
            if not isinstance(validation_hooks, (list, tuple)):
                raise ValueError('"hooks" (in the "validate" section) should '
                                 'be a list of hook specifications.')
            validation_hooks = [EvaluationHook.from_specification(spec)
                                for spec in validation_hooks]
        else:
            validation = None
            best_valid = None
            validation_hooks = None

		# get train_weights, if it is string, set train_weights to be initial weights; if it is a dict, train_weights has 3 elements: initial, best, and last weights
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

		# get train section's checkpoint
        checkpoint = self.data['train'].get('checkpoint')

        if deprecated_checkpoint is not None:
            warnings.warn('"checkpoint" belongs under "train", not under '
                          '"weights".', DeprecationWarning)
            if checkpoint is None:
                checkpoint = deprecated_checkpoint
            else:
                logger.warning('The currently-accepted "checkpoint" will be '
                               'used over the deprecated "checkpoint".')

	    # get file path for initial_weights, best_train weights, best_valid and last_weights
        def expand(x): return os.path.expanduser(os.path.expandvars(x))
        initial_weights, best_train, best_valid, last_weights = [
            expand(x) if x is not None else x for x in
            (initial_weights, best_train, best_valid, last_weights)
        ]

		# build model object using data provider to spec, then build Executor trainer,
        model = self.get_model(provider)
        trainer = self.get_trainer()

		# create the actual training function: 1. if initial_weights is available, then restore the weights to model; 2. store all weights paths, data provider, hooks, checkpoint in a dict named defaults; 3. update this dict with additional unkown named input arguments; 4. train the Executor trainer with defaults dict
        def func(**kwargs):
            """ Trains a model from a pre-packaged specification file.
            """
			# if initial_weights is available, then restore model with this weights
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
                        logger.info('Ignoring missing initial weights: %s. If '
                                    'this is undesireable, set "must_exist" to "yes" '
                                    'in the approriate "weights" section.',
                                    initial_weights)

			# store all weights paths, data provider, hooks, checkpoint in a dict named defaults
            defaults = {
                'provider': provider,
                'validation': validation,
                'stop_when': stop_when,
                'log': log,
                'best_train': best_train,
                'best_valid': best_valid,
                'last_weights': last_weights,
                'training_hooks': training_hooks,
                'validation_hooks': validation_hooks,
                'checkpoint': checkpoint
            }
			# update this list with additional unkown named input arguments
            defaults.update(kwargs)
			# train the Executor trainer with defaults dict
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
        logger.info('Using multiple providers. Since there is no '
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

        def expand(x): return os.path.expanduser(os.path.expandvars(x))
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
                'providers': providers,
                'validating': False,
                'hooks': hooks
            }
            defaults.update(kwargs)
            return trainer.test(**defaults)

        return func

    ###########################################################################
    def get_trainer(self, with_optimizer=True):
        """ get trainer is to Execute with model, loss and optimizer: 1. get a Model object; 2. get a loss object e.g. CategoricalCrossentropy object; 3. get a optimizer object, Adam object; They are no more dict from spec

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

        spec = dict(self.data['train'].get('optimizer', {}))
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

        def expand(x): return os.path.expanduser(os.path.expandvars(x))
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
                'provider': provider,
                'callback': None
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
        """ parse():self.containers<->self._parse_model(
            engine, section, stack, required=True): convert model dicts from spec.data['model'] to a list of containers of all layers: 1. stack: provide scopes for evaulation; 2. section: a string for top level sections, here to be 'model'; 3. make sure 'model' is a key found in spec.data; 4. convert each dict element of spec.data['model'] to a container, parse container with engine, and store all containers in a list; 5. return the list of containers.

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
        logger.info("parse():self.containers<->self._parse_model(engine, section, stack, required=True): convert model dicts from spec.data['model'] to a list of containers of all layers: 1. stack: provide scopes for evaulation; 2. section: a string for top level sections, here to be 'model'; 3. make sure 'model' is a key found in spec.data; 4. for each dict element or entry of spec.data['model'] create a container object, parse container with engine, and store all containers in a list; 5. return the list of containers.")

		# make sure 'model' is a key in spec.data
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

		# take each dict elemement entry from spec.data['model'], then do JinjaEngine.evaluate(entry), then create a Container object based on it, and store it inside containers
        with ScopeStack(engine, stack):
            queue = deque(self.data[key])
            containers = []
            while queue:
                entry = queue.popleft()
                entry = engine.evaluate(entry)
                containers.append(
                    Container.create_container_from_data(entry)
                )
			# each container insider containers are objects like kur.containers.layers.placeholder.Placeholder...; after container.parse(), container.__dict__['_parsed'] is set True, ...['name'] is set 'images'
            for container in containers:
                container.parse(engine)

        return containers

    ###########################################################################
    def parse_source(self, engine, *,
                     source=None, context=None, loaded=None):
        """ __init__():self.data->self.parse_source(engine, source=filename, context=None): 1. read data from a yml file, and its-includes files (recursively); 2. merge the data; 3. returns the merged data.
        """
        logger.info("__init__():self.data->self.parse_source(engine, source=filename, context=None): 1. read data from a yml file, and its-includes files (recursively); 2. merge the data; 3. returns the merged data.")

        def get_id(filename):
            """ Returns a tuple which uniquely identifies a file.

                    This function follows symlinks.
            """
            filename = os.path.expanduser(os.path.expandvars(filename))
            stat = os.stat(filename)
            return (stat.st_dev, stat.st_ino)

        # Process the "loaded" iterable.
        loaded = loaded or set()

		# get filename for yaml
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

        logger.debug('__init__->parse_source(): Parsing source: %s, included by %s.', filename,
                    context or 'top-level')

        if context:
            filename = os.path.join(os.path.dirname(context), filename)
        expanded = os.path.expanduser(os.path.expandvars(filename))
        if not os.path.isfile(expanded):
            raise IOError('Error while parsing source file {}. No such '
                          'source file found: {}. Path was expanded to: {}'.format(
                              context or 'top-level', filename, expanded))

	    # get unique id for the yml file, avoid repeating adding files
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

		# read data from a the main yaml file
        try:
            data = Reader.read_file(expanded)
        except:
            logger.exception('Failed to read file: %s. Check your syntax.',
                             expanded)
            raise

		# get include yml filename
        new_sources = data.pop('include', [])

		# kur.JinjaEngine.evaluate this yml file and return a file path ??
        engine.evaluate(new_sources, recursive=True)

		# get the yml file into a list
        if isinstance(new_sources, str):
            new_sources = [new_sources]
        elif not isinstance(new_sources, (list, tuple)):
            raise ValueError('Error while parsing file: {}. All "include" '
                             'sections must be a single source file or a list of source '
                             'files. Instead we received: {}'.format(
                                 expanded, new_sources))
		# func: read data_new from a new yml file and then merge it with the existing data
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

		# new includes source to be read and merged: can be dict, string or else
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
            kwargs = {'recursive': recursive}

        if source:
            path = os.path.join(os.path.dirname(source), path)
        yield from glob.iglob(path, **kwargs)

    ###########################################################################
    def _parse_templates(self, engine, section, stack):
        """ parse()->_parse_templates(self, engine, section, stack): return the templates dict from spec.data['templates']: 1. section should be a string as 'templates'; 2. 'templates' should be a key in spec.data; 3. spec.data['templates'] should be a dict; 4. keys out of spec.data['templates'] should not be a real container object name or real layer object name; 5. engine object has _scope, _templates, env, state; only env is not Empty; fill _templates with spec.data['templates']; 6. return spec.data['templates']
        """
        logger.info("parse()->_parse_templates(self, engine, section, stack): return the templates dict from spec.data['templates']: 1. section should be a string as 'templates'; 2. 'templates' should be a key in spec.data; 3. spec.data['templates'] should be a dict; 4. keys out of spec.data['templates'] should not be a real container object name or real layer object name; 5. engine object has _scope, _templates, env, state; only env is not Empty; fill _templates with spec.data['templates']; 6. return spec.data['templates']")
		# section should be a string as 'templates'
        if isinstance(section, str):
            section = (section, )

		# 'templates' should be a key in spec.data
        key = None		# Not required, but shuts pylint up.
        for key in section:
            if key in self.data:
                break
        else:
            return None

        if not self.data[key]:
            return None

		# spec.data['templates'] should be a dict
        if not isinstance(self.data[key], dict):
            raise ValueError('Section "{}" should contain a dictionary of '
                             'templates.'.format(key))

		# keys out of spec.data['templates'] should not be a real container object name or real layer object name
        for template in self.data[key]:
            other = Container.get_container_for_name(template)
            if other:
                raise ValueError('Templates cannot override built-in layers. '
                                 'We found a conflict with template "{}".'.format(template))
		# engine object has _scope, _templates, env, state; only env is not Empty; fill _templates with spec.data['templates']
        engine.register_templates(self.data[key])
		# return the templates dict from spec.data
        return self.data[key]

    ###########################################################################
    def _parse_section(self, engine, section, stack, *,
                       required=False, include_key=True, auto_scope=False):
        """ 1. this func assist spec.parse() to add section aliases onto spec.data; 2. Parses a single top-level entry in the Kurfile.

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
        logger.info("")
        # make the selected section to be a tuple
        if isinstance(section, str):
            section = (section, )

        logger.debug('_parse_section(): Parsing Kurfile section: %s', section[0])

        # assign the section name to key
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

        # make a list extra_stack
                # element is a dict, {key : self.data['settings']}
        extra_stack = [{key: self.data[key]}]

        # add self.data['settings'] to extra_stack, as second dict
        if auto_scope:
            extra_stack += [self.data[key]] \
                if isinstance(self.data[key], dict) else []

        # Chek this question and understanding
            #  http://stackoverflow.com/questions/43291756/how-to-understand-scope-and-engine-evaluate-in-parse-section-in-kur

		# add stack as a list of dicts ['host', 'raw', 'parsed', 'filename'] and extra_stack ['settings'] to engine._scope
        with ScopeStack(engine, stack + extra_stack):

            evaluated = engine.evaluate(self.data[key], recursive=True)

        if not include_key:
            if evaluated is not None:
                if not isinstance(evaluated, dict):
                    raise ValueError(
                        'Section "{}" should contain key/value pairs.'
                        .format(key))
                stack.append(evaluated)
        stack.append({key: evaluated})

        # Now, just in case this was aliased, let's make sure we keep our
        # naming scheme consistent.
        for key in section:
            self.data[key] = evaluated

        return evaluated

# EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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
import logging
import shutil
import math
import time
import traceback
import numpy
import tqdm
from ..providers import Provider
from ..utils import get_any_value, CriticalSection, parallelize, Timer
from ..loggers import PersistentLogger
from .hooks import TrainingHook

logger = logging.getLogger(__name__)

# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
###############################################################################
class RetryException(Exception):
	""" Exception class for retrying an operation on a new batch of data.
	"""
	pass

###############################################################################
class Executor:
	""" Class for using models.
	"""

	MAX_RETRIES = 3
	DEFAULT_RETRY_ENABLED = True
	NAN_IS_FATAL = True

	###########################################################################
	def __init__(self, model, loss=None, optimizer=None, auto_retry=None):
		""" Creates a new executor.

			# Arguments

			model: Model instance. The model to train.
			loss: Loss instance. The loss function to use in training/testing.
			optimizer: Optimizer instance. The optimizer to use in training.
		"""
		logger.critical("(self, model, loss=None, optimizer=None, auto_retry=None): \n\nCreates a new executor object: \n1. add model object as its attr; \n2. add loss object as its attr; \n3. add optimizer object as its attr; \n4. set its attr auto_retry as True \n\n")

		self.model = model
		self.loss = loss
		self.optimizer = optimizer

		if auto_retry is None:
			auto_retry = self.DEFAULT_RETRY_ENABLED
		self.auto_retry = auto_retry

	###########################################################################
	def compile(self, target=None, recompile=False, with_provider=None,
		**kwargs):
		""" Compiles a model. 1. set target to train, test or evaluate; 2. make sure Executor.model is built before compilation; 3. Compiles a model using keras backend; 4. dive into keras.backend.compile, additional dataset is handled

			This generates a backend-specific representation of the model,
			suitable for training.

			# Arguments

			recompile: bool (default: False). If the model has already been
				compiled, it is not compiled again unless this flag is True.
			with_provider: Provider instance or None (default: None). If you
				want to merge the model's auxiliary data sources into your
				provider, you can specify the Provider instance here.

			# Return value

			None
		"""
		logger.critical("(self, target=None, recompile=False, with_provider=None, **kwargs): \nCompile a model \n\n1. If section is not selected, given loss and optimizer are not None, select train section to build; \n2. make sure the model is built before compilation; \n3. compile model with backend using model, loss, optimizer objects using kur.backend.keras_backend \n\n")
		# If section is not selected, given loss and optimizer are not None, select train section to build
		if target is None:
			if self.loss is None and self.optimizer is None:
				target = 'evaluate'
			elif self.optimizer is None:
				target = 'test'
			else:
				target = 'train'

		if not recompile:
			if self.model.compiled is not None \
				and target in self.model.compiled:
				return

		# make sure the model is built before compilation
		if not self.model.is_built():
			logger.warning('This model has never been built before. We are '
				'going to try to build it now. But the model should always be '
				'built with Model.build() before trying to compile it, just '
				'to ensure that everything has been parsed as you expect.')
			if with_provider is not None:
				self.model.register_provider(with_provider)
			self.model.build()

		logger.trace('Recompiling the model.')
		# compile model with backend using model, loss, optimizer objects
		self.model.backend.compile(
			model=self.model,
			loss=self.loss if target != 'evaluate' else None,
			optimizer=None if target != 'train' else self.optimizer,
			blocking=True,
			**kwargs
		)

		if with_provider is not None:
			if isinstance(with_provider, Provider):
				self.model.supplement_provider(with_provider)
			elif isinstance(with_provider, dict):
				for provider in with_provider.values():
					self.model.supplement_provider(provider)
			else:
				raise ValueError('Unexpected provider type: {}'
					.format(with_provider))

	###########################################################################
	def test(self, providers, validating=False, hooks=None, step=False):
		""" Tests/validates the model on some data.

			# Arguments

			providers: dict. The keys are provider names, and the values are
				Provider instances. The data provider which serves the data to
				be evaluated on.
			validating: bool (default: False). If False, the console output
				refers to this process as "testing"; otherwise, it is referred
				to as "validating."

			# Return value

			The average loss across the validation set.
		"""

		self.compile('test', with_provider=providers)

		loss = {}
		counts = {}
		for k, provider in providers.items():
			n_entries, test_loss = self.test_with_provider(
				provider,
				name=k if len(providers) > 1 else None,
				validating=validating,
				hooks=hooks,
				step=step
			)
			#if 'total' not in test_loss:
			#	test_loss['total'] = sum(test_loss.values())
			loss[k] = test_loss
			counts[k] = n_entries

		total_count = sum(counts.values())
		average = {}
		for provider_name, loss_dict in loss.items():
			for branch_name, loss_value in loss_dict.items():
				weight = counts[provider_name] / total_count
				contribution = loss_value * weight
				if branch_name not in average:
					average[branch_name] = 0
				average[branch_name] += contribution

		if len(providers) > 1:
			logger.info(
				'Overall %s loss: %.3f',
				'validation' if validating else 'testing',
				sum(average.values())
			)

		return average, loss

	###########################################################################
	def test_with_provider(self, provider, *, name=None,
		validating=False, hooks=None, step=False):
		""" Tests/validates the model on a single provider.
		"""

		if validating:
			desc = ('Validating', 'Validation')
		else:
			desc = ('Testing', 'Test')

		# Create progress bar
		test_loss = None
		n_entries = 0
		first_batch = None
		test_func = self.retry(
			self.model.backend.test,
			self.auto_retry
		)
		with tqdm.tqdm(
					total=len(provider),
					unit='samples',
					desc='{}{}, loss=N/A'.format(
						desc[0],
						' "{}"'.format(name) if name else ''
					)
				) as pbar:

			# Present each batch to the network.
			for num_batches, batch in parallelize(enumerate(provider)):
				if step:
					self.do_step('Test', num_batches, batch)

				try:
					prediction, batch_loss = test_func(
						model=self.model,
						data=batch
					)
				except RetryException:
					continue

				if step and logger.isEnabledFor(logging.DEBUG):
					print(prediction)

				if first_batch is None:
					first_batch = (prediction, batch)

				batch_size = len(get_any_value(batch))

				#batch_loss = loss if isinstance(loss, float) \
				#	else sum(loss.values())

				new_entries = n_entries + batch_size

				if test_loss is None:
					test_loss = batch_loss
				else:
					test_loss = {
						k : v * (n_entries / new_entries) + \
							batch_loss[k] * (batch_size / new_entries)
						for k, v in test_loss.items()
					}
				#avg_loss = avg_loss * (n_entries / new_entries) + \
				#	batch_loss * (batch_size / new_entries)
				n_entries = new_entries

				# Update the progress bar
				pbar.set_description('{}{}, loss={:.3f}'.format(
					desc[0],
					' "{}"'.format(name) if name else '',
					sum(test_loss.values())
				))
				pbar.update(batch_size)

		if not n_entries:
			logger.warning('No data provided to validation/testing system.')
			return None

		logger.info('%s loss: %s%.3f',
			desc[1],
			'"{}"='.format(name) if name else '',
			sum(test_loss.values())
		)

		if hooks and first_batch is not None:
			prediction, batch = first_batch
			prev = first_batch
			for hook in hooks:
				new_prev = hook.apply(prev, first_batch, self.model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev

		return n_entries, test_loss

	###########################################################################
	def train(self, *args, last_weights=None, log=None, training_hooks=None,
		**kwargs):
		""" Trains the model on some data.

			This is the public entry point for training. It wraps the business
			logic so that it can handle error conditions.
		"""
		logger.critical("\n\nNow a model's weights is restored, Training can start. \nDive into result = self.wrapped_train(*args, log=log, training_hooks=training_hooks, **kwargs)\n\nSee all inputs of wrapped_train() here ")

		pprint(getfullargspec(self.wrapped_train))
		reason = 'unknown'
		try:
			result = self.wrapped_train(
				*args,
				log=log,
				training_hooks=training_hooks,
				**kwargs
			)
		except (KeyboardInterrupt, Exception) as exc:
			logger.exception('Exception raised during training.')
			reason = traceback.format_exception_only(type(exc), exc)[0].strip()
			raise

		# except may be escaped, but else and finally must be run.
		else:
			reason = 'success'
			return result
		finally:
			logger.warning("\n\nSave the last weights into the folder of last weights: \n{}\n\n".format(last_weights))
			if last_weights is not None:
				logger.warning('Saving most recent weights: %s\nUsing with CriticalSection():self.model.save(last_weights)', last_weights)
				with CriticalSection():
					self.model.save(last_weights)

			logger.warning("\n\nWhat does log.flush() do?\n\n")
			if log is not None:
				log.flush()

			logger.warning("\n\nCreate plots\n\n")
			if training_hooks:
				for hook in training_hooks:
					hook.notify(
						TrainingHook.TRAINING_END,
						log=log,
						info={'Reason' : reason}
					)

	###########################################################################
	def wrapped_train(self, provider, *, validation=None, stop_when=None,
		log=None, best_train=None, best_valid=None, training_hooks=None,
		validation_hooks=None, checkpoint=None, step=False):
		""" Trains the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the
				data to be trained on.
			validation: Provider instance or None (default: None). The data
				provider which serves validation data.
			stop_when: dict or None (default: None). Stopping criteria.
			log: Log instance or None (default: None). The logger to save
				training statistics with.

			# Return value

			None
		"""
		logger.critical("\n\nwrapped_train(self, provider, *, validation=None, stop_when=None,log=None, best_train=None, best_valid=None, training_hooks=None,validation_hooks=None, checkpoint=None, step=False)\n\nIt has included a lot of functions\n\nExecutes a validation run: \nrun_validation(num_batches=None)\n\nSaves the current model weights.\nsave_or_copy_weights(target)\n\nCalculates training loss and saves if necessary.\nrun_posttrain(n_entries, train_loss)\n\nExecutes the training hooks, if necessary.\nrun_training_hooks(cur_train_loss, validation_loss, status)\n\nPretty-prints a number of seconds.\nwrite_time(title, seconds)\n\nPrints the current timer values.\nprint_times()\n\nRuns the checkpoint triggers, if necessary.\nrun_checkpoint(*triggers, allow_validation=True)")
		

		#######################################################################
		def run_validation(num_batches=None):
			""" Executes a validation run.
			"""
			if validation is None:
				return None

			nonlocal best_valid_loss

			timers['validate'].resume()

			# Continue with a validation run.
			previous_num_batches = {}
			try:
				if num_batches is not None:
					for provider in validation.values():
						if hasattr(provider, 'num_batches'):
							previous_num_batches[id(provider)] = \
								provider.num_batches
							provider.num_batches = num_batches

				average_loss, validation_loss = self.test(
					providers=validation,
					validating=True,
					hooks=validation_hooks
				)
			finally:
				if num_batches is not None:
					for provider in validation.values():
						if hasattr(provider, 'num_batches'):
							provider.num_batches = \
								previous_num_batches[id(provider)]

			if validation_loss is None:
				timers['validate'].pause()
				return None

			cur_validation_loss = sum(average_loss.values())
			validation_loss[None] = average_loss
			logger.debug('Current validation loss: %.3f', cur_validation_loss)

			if best_valid is not None:
				if best_valid_loss is None or \
						cur_validation_loss < best_valid_loss:
					logger.debug(
						'Saving best historical validation weights: %s',
						best_valid
					)
					best_valid_loss = cur_validation_loss
					save_or_copy_weights(best_valid)

			if log is not None:
				log.log_validation(validation_loss, 'loss', clocks=timers)

			timers['validate'].pause()
			return validation_loss

		#######################################################################

		def save_or_copy_weights(target):
			""" Saves the current model weights.
			"""
			nonlocal saved_recent

			if saved_recent is None:
				logger.trace('Saving weights to: %s', target)
				with CriticalSection():
					self.model.save(target)
				saved_recent = target
			elif not os.path.exists(saved_recent):
				logger.warning('Recently saved weight file seems to have '
					'vanished: %s', saved_recent)
				saved_recent = None
				save_or_copy_weights(target)
			elif os.path.exists(target) and \
					os.path.samefile(target, saved_recent):
				logger.trace('Recent weight file seems the same as the '
					'soon-to-be-saved file. Skipping: %s', target)
			else:
				logger.trace('Copying weights from: %s', saved_recent)
				with CriticalSection():
					shutil.rmtree(target, ignore_errors=True)
					shutil.copytree(saved_recent, target)

		#######################################################################

		def run_posttrain(n_entries, train_loss):
			""" Calculates training loss and saves if necessary.

				Read-only non-locals:
					n_entries, train_loss, best_train, log
				Read-write non-locals:
					best_train_loss
			"""

			nonlocal best_train_loss
			if not n_entries:
				logger.warning('No data provided to training loop.')
				return None

			# get current training loss of an epoch
			cur_train_loss = sum(train_loss.values())
			logger.info('Training loss: %.3f', cur_train_loss)

			# if best_train folder exist
			if best_train is not None:
				# if best_train_loss is None, set best_train_loss = cur_train_loss
				if best_train_loss is None or \
					cur_train_loss < best_train_loss:

					logger.debug('Saving best historical training weights: '
						'%s', best_train)
					best_train_loss = cur_train_loss
					# save current training loss into cifar.best.train.w folder
					save_or_copy_weights(best_train)

			# if log is available
			if log is not None:
				# update Log training statistics after an epoch
				print("\nBefore log.log_training: inside log.__dict__ we got")
				pprint(log.__dict__)

				log.log_training(train_loss, 'loss', clocks=timers)

				print("\n\nafter log.log_training: inside log.__dict__ we got\n")
				pprint(log.__dict__)
				print("\n\n")

			return cur_train_loss

		#######################################################################


		def run_training_hooks(cur_train_loss, validation_loss, status):
			""" Executes the training hooks, if necessary.

				Read-only non-locals:
					training_hooks, epoch, epochs, validation_loss
			"""
			if not training_hooks:
				return
			info = {
				'epoch' : epoch+1,
				'total_epochs' : epochs,
				'Training loss' : cur_train_loss
			}
			if validation is not None:
				info['Validation loss'] = validation_loss
			for hook in training_hooks:
				hook.notify(
					status,
					log=log,
					info=info
				)

		#######################################################################

		def write_time(title, seconds):
			""" Pretty-prints a number of seconds.
			"""
			seconds = int(seconds)
			minutes, seconds = divmod(seconds, 60)
			hours, minutes = divmod(minutes, 60)
			tqdm.tqdm.write('{}: {:02d}h {:02d}m {:02d}s'.format(
				title, hours, minutes, seconds
			))

		#######################################################################


		def print_times():
			""" Prints the current timer values.
			"""
			write_time('     Total wall-clock time', timers['all'].get())
			write_time('  Training wall-clock time', timers['train'].get())
			if validation is not None:
				write_time('Validation wall-clock time',
					timers['validate'].get())
			write_time('     Batch wall-clock time', timers['batch'].get())

		#######################################################################


		def run_checkpoint(*triggers, allow_validation=True):
			""" Runs the checkpoint triggers, if necessary.
			"""
			# try checkpoint is not None sometime
			nonlocal last_checkpoint

			if checkpoint is None:
				return False

			timers['train'].pause()
			for k in triggers:
				if k not in checkpoint:
					continue
				if session[k] - last_checkpoint[k] >= checkpoint[k]:
					# We need a checkpoint

					# Save the file if necessary.
					if checkpoint['path']:
						tqdm.tqdm.write('Checkpoint...')
						logger.debug('Making checkpoint backup: %s',
							checkpoint['path'])
						save_or_copy_weights(checkpoint['path'])

					# Validate if necessary.
					if checkpoint.get('validation', False) \
							and allow_validation:
						if isinstance(checkpoint['validation'], bool):
							num_batches = None
						else:
							num_batches = checkpoint['validation']
						val_loss = run_validation(num_batches)
						run_training_hooks(None, val_loss,
							TrainingHook.VALIDATION_END)

					last_checkpoint = session.copy()

					timers['train'].resume()
					return True

			timers['train'].resume()
			return False

		#######################################################################
		logger.warning("\n\nCreate 4 timers: batch, train, validate, all\n\n")

		timers = {
			'batch' : Timer(started=False),
			'train' : Timer(started=False),
			'validate' : Timer(started=False),
			'all' : Timer(started=False)
		}

		#######################################################################

		logger.critical("\n\nhandles when checkpoint is dic, str, or else or None\n\nMake sure the values inside checkpoint are expected\n\n")
		# handles when checkpoint is dic, str, or else or None
		# Process checkpoint requirements
		if isinstance(checkpoint, dict):
			if 'path' not in checkpoint:
				checkpoint['path'] = 'checkpoint'

			found = False
			for k in ('epochs', 'batches', 'samples', 'minutes'):
				if k in checkpoint:
					if not isinstance(checkpoint[k], int):
						raise ValueError('Expected "{}" key in "checkpoint" '
							'to be an integer. Received: {}'.format(k,
							checkpoint[k]))
					found = True

			if not found:
				checkpoint['epochs'] = 1

		elif isinstance(checkpoint, str):
			checkpoint = {
				'path' : checkpoint,
				'epochs' : 1
			}
		elif checkpoint is not None:
			raise ValueError('Unknown format for "checkpoint". Expected a '
				'single file or a dictionary. Instead we received: {}'
				.format(checkpoint))

		#######################################################################


		# Parse logs
		logger.critical("\n\nif log is not available or non-persistent type, there is no historical loss or best_valid_loss nor best_train_loss\n\n")
		# if log is not available or non-persistent type, there is no best_valid_loss nor best_train_loss
		if log is None:
			logger.info('No log specified, so no historical loss information '
				'is available.')
			best_train_loss = best_valid_loss = None
		elif not isinstance(log, PersistentLogger):
			logger.info('Log type is non-persistent, so no historical loss '
				'information is available.')
			best_train_loss = best_valid_loss = None

			logger.critical("\n\nIf log is available, load from files the historical `best_train_loss` and `best_valid_loss` using `log.get_best_training_loss()` and `log.get_best_validation_loss()`; \n\n")
		else:
			best_train_loss = log.get_best_training_loss()
			if best_train_loss is not None:
				logger.warning('Best historical training loss: %.3f',
					best_train_loss)
			else:
				logger.warning('No historical training loss available from logs.')

			best_valid_loss = log.get_best_validation_loss()
			if best_valid_loss is not None:
				logger.warning('Best historical validation loss: %.3f',
					best_valid_loss)
			else:
				logger.warning(
					'No historical validation loss available from logs.')

			logger.critical("\n\nget clocks using `log.get_clocks()` and `print_times()`\n\n")
			clocks = log.get_clocks()
			if clocks:
				for k, v in clocks.items():
					if k in timers:
						timers[k].reset(v)
				print_times()

		#######################################################################

		# Parse desired number of epochs
		logger.critical("\n\nget previous trained epochs `completed_num_epochs` using `log.get_number_of_epochs()`\n\n")
		completed_epochs = log.get_number_of_epochs() if log else 0
		if not completed_epochs:
			logger.warning('No previous epochs.')
		else:
			logger.warning('Restarting from epoch %d.', completed_epochs+1)

		#######################################################################

		logger.critical("\n\nprovide two stop modes (additional or total)\n\nget `mode` from `stop_when.get('mode', valid_modes[0])`\n\nif mode is set to 'total' but `log` is `None`, then set mode to `additional`\n\n")
		# Parse the stopping criterion mode.
		# provide two stop modes (additional or total)
		valid_modes = ('additional', 'total')
		# get mode from stop_when.get('mode', valid_modes[0]), if stop_when is not available, use valid_modes[0]
		mode = stop_when.get('mode', valid_modes[0])
		print("mode is set to {} inside stop_when['mode']".format(mode))
		# make sure no other modes are allowed
		if mode not in valid_modes:
			raise ValueError('"mode" in "stop_when" must be one of: {}. '
				'Instead, we received: {}.'.format(', '.join(valid_modes),
				mode))
		# if mode is set to 'total' but log is None, then set mode to additional
		if mode == 'total' and log is None:
			logger.warning('The epoch specification has "mode" set to "%s". '
			'This mode requires a log to be used correctly. Kur will proceed '
			'as if "mode" were "%s".', mode, valid_modes[0])
			mode = valid_modes[0]

		#######################################################################

		logger.critical("\n\nExtract epochs from `spec.data['train']['stop_when']['epochs']`\n\n")
		# Parse "epoch" stopping criterion.
		# set stop epochs: get epochs for stopping using stop_when.get('epochs')
		epochs = stop_when.get('epochs')

		logger.critical("\n\nIf epochs is a string from ('inf', 'all', 'infinite', 'infinity'), set it None\n\n")
		# if epochs set by user as infinity alike, set epochs = None
		if epochs in ('inf', 'all', 'infinite', 'infinity'):
			epochs = None

		logger.critical("\n\nmake sure epochs is either int or None\n\n")
		# make sure epochs is either int or None
		if not isinstance(epochs, (int, type(None))):
			raise ValueError('Expected "epochs" to be a None or an integer. '
				'Instead, we received: {}.'.format(epochs))

		logger.critical("\n\nIf mode is additional, epochs = epochs + completed_epochs\n\n(If mode is total, and) completed_epochs >= epochs, stopping criteria is met\n\n ")
		# if epochs is available, and mode is additional, the add up epochs with completed_epochs
		if epochs is not None:
			if mode == 'additional':
				epochs += completed_epochs
				logger.warning("total epochs accumulated after training: {}\n\n".format(epochs))
			# stop training when completed_epochs >= epochs:
			if completed_epochs >= epochs:
				logger.warning("If previous trained epochs > epochs, then Epoch stopping-criterion met. This indicate total mode, so total epochs : {}\n\n".format(completed_epochs))
				return

		#######################################################################
		logger.critical("\n\nExtract clock setting from stop_when, see pprint(clock)\n\n")
		# Parse "elapsed" stopping criterion.
		# Set clock for stop_when:
		# There are 4 timer objects (all can be called), the default is 'all' object
		default_time_keeper = 'all'
		# get clock setting from 'clock = stop_when.get('elapsed')'
		clock = stop_when.get('elapsed')
		pprint(clock)
		print("\n\n")

		logger.critical("\n\nExtract the timer object, using `time_keeper = clock.get('clock', default_time_keeper)`\n\n")
		# consider when clock is a dict:
		if isinstance(clock, dict):
			# get timer object using clock.get('clock', default_time_keeper)
			time_keeper = clock.get('clock', default_time_keeper)
			if time_keeper not in timers:
				raise ValueError('Invalid value for '
					'"stop_when.elapsed.clock". Must be one of: {}. Received: '
					'{}'.format(', '.join(timers), time_keeper))
			print("time_keeper is {}\n\n".format(time_keeper))

			logger.critical("get how long user want model to run in terms of minutes, using clock['mintes'|'hours'|'days']*multiplier(1|60|1440) to add up toward the total minutes of training time, save into variable clock_time\n\nHandle clock when it is dict, int, float, str(inf, all, infinite, infinity)\n\n")
			# get how long user want model to run in terms of minutes, using clock['mintes'|'hours'|'days']*multiplier(1|60|1440) to add up toward the total minutes of training time, save into variable clock_time
			clock_time = 0
			for multiplier, value in (
				(1, 'minutes'), (60, 'hours'), (1440, 'days')
			):
				if value not in clock or not clock[value]:
					continue
				if not isinstance(clock[value], (int, float)):
					raise ValueError('Invalid value for "stop_when.clock.{}": '
						'{}'.format(value, clock[value]))
				clock_time += clock[value] * multiplier

		# When clock is int or float, clock refer to num_minutes to run, timer object is 'all'
		elif isinstance(clock, (int, float)):
			clock_time = clock  # Defaults to minutes.
			time_keeper = 'default_time_keeper'
		# when clock is string and clock is about infinitiy
		elif isinstance(clock, str) and clock in \
			('inf', 'all', 'infinite', 'infinity'):
			# set clock=None for training without time constraint
			clock = None
		elif clock:
			raise ValueError('Invalid value for "stop_when.elapsed". Should '
				'be a dictionary or numeric. Received: {}'.format(clock))

		print("Total time for training: clock_time is {}\n\n".format(clock_time))

		# clock_time can't be negative or 0
		if clock:
			if clock_time <= 0:
				raise ValueError('"stop_when.elapsed" resolved to a '
					'non-positive value: {}'.format(clock_time))

			clock = {
				# clock_time eventually use second unit
				'seconds' : clock_time*60,
				# timer object is picked
				'timer' : timers[time_keeper],
				# initialize mark to 0
				'mark' : 0
			}


			# set the starting time and store in clock['mark'], when mode == 'additional', set clock['mark'] += clock['timer']()
			if mode == 'additional':
				clock['mark'] += clock['timer']()

			logger.critical("\n\nIf (clock['timer']() - clock['mark']) > clock['seconds'] is true, then it indicates `mode = total` and stopping criteria is met\n\n")

			if (clock['timer']() - clock['mark']) > clock['seconds']:
				print('Elapsed-time stopping criterion met.')
				return


		#######################################################################
		logger.critical("\n\nsaved_recent is the latest saved weights file; \n\n create a session dict (epochs, batches, samples, minutes) and assign a copy to last_checkpoint; \n\n retry the training function for at most 3 times;\n\n")

		saved_recent = None

		# create a session and make it the last_checkpoint
		session = {
			'epochs' : 0,
			'batches' : 0,
			'samples' : 0,
			'minutes' : time.perf_counter() / 60
		}
		last_checkpoint = session.copy()

		# ???
		epoch = completed_epochs - 1

		# create a function to retry training using keras_backend.train for at most 3 times
		train_func = self.retry(
			self.model.backend.train,
			self.auto_retry
		)


		#######################################################################
		# Prepare to train

		logger.critical("\n\ncompile Executor trainer on train section: save weights, test 2 data samples, restore weights\n\n")
		self.compile('train', with_provider=provider)

		logger.critical("\n\noutput some log info on provider; \n\ncreate the plot; \n\n")
		# Prints debug information about the sources in this provider
		provider.source_shapes()

		if training_hooks:
			for hook in training_hooks:
				# Notifies the hook that the epoch has ended.
				hook.notify(
					TrainingHook.TRAINING_START,
					log=log
				)

		all_done = False


		#######################################################################
		logger.critical("\n\n. Main training loop until all_done is set True or just return???: \n\n resume timer using timers['all'].resume(): Starts a previously paused timer, this has no effect if the timer was not paused; \n\n count epoch, and stop looping or training when total_num_epochs reached; \n\n")

		# Main training loop.
		timers['all'].resume()
		while not all_done:
			epoch += 1
			if epochs is not None and epoch >= epochs:
				print('Completed {} epochs.'.format(epochs))
				break

			print()

			###################################################################

			# START: Train one epoch
			logger.critical("\n\nResume timer for training\n\n")
			timers['train'].resume()

			# Create progress bar
			logger.critical("\n\nCreate a progress bar to describe the displaying details\n\n")
			train_loss = None
			n_entries = 0
			with tqdm.tqdm(
						total=len(provider),
						unit='samples',
						desc='Epoch {}/{}, loss=N/A'
							.format(epoch+1, epochs or 'inf')
					) as pbar:

				# Present each batch to the network.
				for num_batches, batch in parallelize(enumerate(provider)):

					# The loss averaged over this batch.
					logger.critical("\n\nTraining on a batch...\n\nUse step feature or not \n\nResume timer for a batch\n\nCalc prediction, and batch_loss of this batch, and then pause batch timer\n\n")
					if step:
						# Wait for user input before running a single batch of data.
						self.do_step(
							'Train, Epoch {}'.format(session['epochs']+1),
							num_batches, batch)
					print("Use step feature: {}\n\n".format(step))


					# resume time for batch
					timers['batch'].resume()
					try:
						# train on one batch to get prediction and batch_loss, using keras_backend.train, and inside using run_batch
						prediction, batch_loss = train_func(
							model=self.model, data=batch)

						logger.warning("\n\nprediction.key: %s\n", prediction.keys())
						for k, v in prediction.items():
							print("prediction array shape: {}".format(v.shape))
							print("the first row of the array: \n{}\n\n".format(v[0]))

					except RetryException:
						continue
					finally:
						# pause timer on training batches
						timers['batch'].pause()


					# if args.step is true, and logging.DEBUG mode is on, then print out prediction
					if step and logger.isEnabledFor(logging.DEBUG):
						print(prediction)

					# We just modified the weights. Invalidate the name of the last weight file.
					saved_recent = None

					logger.warning('\n\nFinished training on batch.\n\n')

					logger.critical("\n\nUpdate log information for this batch \n\nCheck inside log before and after update log info\n\n")
					# get the num_samples
					batch_size = len(get_any_value(batch))

					if log is not None:
						# Log training information after a batch.
						print("Before training this batch, inside log we got: \n")
						pprint(log.__dict__)
						log.log_batch(batch_size, batch_loss, 'loss',
							clocks=timers)
						print("\n\nAfter training this batch, inside log we got: \n")
						pprint(log.__dict__)
						print("\n\n")

					logger.critical("\n\nUpdate session for this batch\n\nCheck insider session before and after session update\n\n ")
					# Update our session statistics.
					print("\Before training this batch, inside session we got: \n")
					pprint(session)
					session['batches'] += 1
					session['samples'] += batch_size
					session['minutes'] = time.perf_counter() / 60
					print("\n\nAfter training this batch, inside session we got: \n")
					pprint(session)
					print("\n\n")

					logger.critical("\n\nRun checkpoints for true or false, if true, print out wall time\n\n")
					# If there is no checkpoint, just return False
					if run_checkpoint('samples', 'batches', 'minutes', allow_validation=True):
						print_times()

					# num_entries processed so far in this epoch
					new_entries = n_entries + batch_size

					logger.critical("\n\nGet average loss for a batch/a sample\n\n")
					# Average the per-batch loss across training.
					# This will give us our average "training loss".
					if train_loss is None:
						train_loss = batch_loss
					else:
						# if train_loss is available, get average_loss per entry or per batch????
						train_loss = {
							k : v * (n_entries / new_entries) + \
								batch_loss[k] * (batch_size / new_entries)
							for k, v in train_loss.items()
						}
					print("train_loss on average batch or sample: {}\n\n".format(train_loss))

					n_entries = new_entries

					logger.critical("\n\nIf clock for train is set and met, stop training\n\n")
					if clock and clock['seconds'] < \
							(clock['timer'].get() - clock['mark']):
						tqdm.tqdm.write('Timer expired. Finishing up '
							'training.')
						all_done = True
						break


					logger.critical("\n\nUpdate the progress bar with the current loss of an epoch\n\n")

					pbar.set_description('Epoch {}/{}, loss={:.3f}'.format(
						epoch+1, epochs or 'inf', sum(train_loss.values())
					))
					pbar.update(batch_size)

					logger.critical("\n\nmake sure batch_loss values are not Nan, Inf\n\n")

					for k, v in batch_loss.items():
						if math.isnan(v):
							logger.error('Received NaN loss value for '
								'model output "%s". Make sure that your '
								'inputs are all normalized and that the '
								'learning rate is not too high. Sometimes '
								'different algorithms/implementations '
								'work better than others, so you can try '
								'switching optimizers or backend.', k)
							if self.NAN_IS_FATAL:
								raise ValueError('Model loss is NaN.')
						elif math.isinf(v):
							logger.error('Received infinite loss value for '
								'model output "%s". Make sure that your '
								'learning rate is not too high, and that you '
								'clip your gradients.', k)
							if self.NAN_IS_FATAL:
								raise ValueError('Model loss is infinite.')



			logger.critical("\n\npause timer for training (for epoch), end of an epoch training\n\n")

			timers['train'].pause()
			# END: Train one epoch
			###################################################################

			logger.critical("\n\nUpdate our session statistics.\n\nIncrement epochs by 1\n\nrun_checkpoint('epochs', allow_validation=False)\n\nCheck to see what our current training loss is.\n\nCalc validation_loss = run_validation()\n\nrun training hooks\n\nprint wall time\n\n")
			session['epochs'] += 1

			# Checkpoint if necessary
			run_checkpoint('epochs', allow_validation=False)

			# Check to see what our current training loss is.
			cur_train_loss = run_posttrain(n_entries, train_loss)

			# Validate
			validation_loss = run_validation()

			# Execute training hooks.
			run_training_hooks(
				cur_train_loss,
				validation_loss,
				status=TrainingHook.EPOCH_END
			)

			print_times()


	###########################################################################
	def evaluate(self, provider, callback=None, step=False):
		""" Evaluates the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the
				data to be evaluated.
			callback: function or None. If not None, the callback is called
				after each evaluation batch and is passed two parameters:
				`predicted` and `truth`, where `predicted` is the model output
				and `truth` is the ground truth data (if provided by
				`provider`; otherwise, `truth` is set to `None`).

			# Return value

			If `callback` is None, then this returns a tuple `(predicted,
			truth)`, where `predicted` is a dictionary whose keys are the names
			of the output nodes of the model, and whose respective values are
			arrays of predictions (one row per input sample). If the provider
			provides ground truth information, then `truth` has a similar
			structure to `predicted`; if ground truth information is not
			available, then `truth` is None.

			Otherwise, if `callback` is not None, this returns None.
		"""

		self.compile('evaluate', with_provider=provider)

		result = None
		truth = None
		has_truth = None
		total = len(provider)
		n_entries = 0

		#######################################################################
		def store_batch_unknown(batch, evaluated, batch_size):
			""" Saves the batch if we do not know how many entries to expect.
			"""
			nonlocal truth, result

			# We don't know how many entries there will be.
			if result is None:
				# This is our first batch.
				result = {k : [] for k in self.model.outputs}
			for k, v in evaluated.items():
				result[k].extend(v)

			if has_truth:
				if truth is None:
					truth = {k : [] for k in self.model.outputs}
				for k in truth:
					truth[k].extend(batch[k])

		#######################################################################
		def store_batch_known(batch, evaluated, batch_size):
			""" Saves the batch if we know how many entries to expect.
			"""
			nonlocal truth, result

			# We know how many entries there will be.
			if result is None:
				# This is our first batch.
				result = {k : [None]*total for k in evaluated}
			for k, v in evaluated.items():
				result[k][n_entries:(n_entries+batch_size)] = v[:]

			if has_truth:
				if truth is None:
					truth = {k : [None]*total for k in evaluated}
				for k in truth:
					truth[k][n_entries:(n_entries+batch_size)] = batch[k][:]

		store_batch = store_batch_unknown if total is None \
			else store_batch_known

		eval_func = self.retry(
			self.model.backend.evaluate,
			self.auto_retry
		)

		with tqdm.tqdm(
					total=total,
					unit='samples',
					desc='Evaluating'
				) as pbar:

			for num_batches, batch in parallelize(enumerate(provider)):

				if step:
					self.do_step('Evaluate', num_batches, batch)

				try:
					evaluated, _ = eval_func(model=self.model, data=batch)
				except RetryException:
					continue

				if step and logger.isEnabledFor(logging.DEBUG):
					print(evaluated)

				batch_size = len(get_any_value(batch))

				# Check to see if we have truth data available.
				if has_truth is None:
					has_truth = all(k in batch for k in self.model.outputs)

				if callback is None:
					# There is no callback. We need to hang on to everything.
					store_batch(batch, evaluated, batch_size)
				else:
					callback(evaluated, truth)

				n_entries += batch_size
				pbar.update(batch_size)

		if callback is not None:
			return

		for data in (result, truth):
			if data is not None:
				for k, v in data.items():
					data[k] = numpy.array(v)

		if truth is not None:
			for k, v in truth.items():
				result[k] = numpy.reshape(result[k], v.shape)

		return result, truth

	###########################################################################
	def do_step(self, what, num_batches, batch):
		""" Wait for user input before running a single batch of data.
		"""
		print('{}, Batch {}:'.format(what, num_batches+1))
		if logger.isEnabledFor(logging.DEBUG):
			for k, v in batch.items():
				print('{} {}: {}'.format(
					k,
					v.shape if hasattr(v, 'shape') else \
						'(list, {} entries)'.format(len(v)),
					v
				))
		input('Press ENTER to continue...')

	###########################################################################
	def retry(self, func, enabled=True):
		""" Creates a wrapper that implements some retry semantics.
		"""

		def try_func(*args, **kwargs):
			""" Wraps a function with some retry logic.
			"""
			try:
				result = func(*args, **kwargs)

			# Catch Exception so that we don't catch KeyboardInterrupt.
			except Exception:
				if not try_func.enabled:
					raise

				try_func.counter += 1
				if try_func.counter > Executor.MAX_RETRIES:
					logger.exception(
						'Failed to execute on batch. No more retries.')
					raise
				logger.exception('Failed to execute on batch. Tolerating up '
					'to %d more consecutive failures.',
					Executor.MAX_RETRIES - try_func.counter)
				raise RetryException
			else:
				try_func.counter = 0
				return result
		try_func.counter = 0
		try_func.enabled = enabled

		return try_func

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

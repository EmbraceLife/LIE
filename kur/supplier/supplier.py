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

import logging

from ..utils import get_subclasses, get_any_value
from ..sources import StackSource

logger = logging.getLogger(__name__)


from ..utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())
###############################################################################
class Supplier:
	""" Base class for all suppliers.

		Suppliers are essentially factories for Sources. A single data file may
		have multiple data columns available, and a Supplier is responsible for
		parsing out the individual Sources.
	"""

	###########################################################################
	@staticmethod
	def from_specification(spec, kurfile=None):
		""" Creates a new Supplier from a specification.
		"""
		logger.info("\n\nfrom_specification(spec, kurfile=None): \n\n create data supplier object from kurfile dict \n\n1. make sure spec is a dict or entry from upper function; \n\n2. get data supplier name from spec: supplier_name; \n\n3. get entry dict description for this data supplier: params; \n\n4. create the data supplier class by its name, and then instantiate the supplier object with kurfile object and params; \n\n5. return this data supplier \n\n")

		if not isinstance(spec, dict):
			raise ValueError('Each element of the "input" list must be a '
				'dictionary.')

		supplier_name_get = """
candidates = set(
	cls.get_name() for cls in Supplier.get_all_suppliers()
) & set(spec.keys())
		"""
		logger.info("\n\nFind the Supplier class name from the data specified in kurfile\n\n%s\n\n", supplier_name_get)
		candidates = set(
			cls.get_name() for cls in Supplier.get_all_suppliers()
		) & set(spec.keys())

		if not candidates:
			raise ValueError('Missing the key naming the Supplier type from '
				'an element of the "input" list. Valid keys are: {}'.format(
				', '.join(
					cls.get_name()
					for cls in Supplier.get_all_suppliers()
				)
			))

		if len(candidates) > 1:
			raise ValueError('Ambiguous supplier type in an element of the '
				'"input" list. Exactly one of the following keys must be '
				'present: {}'.format(', '.join(candidates)))

		create_supplier_obj = """
name = candidates.pop()
params = spec[name]

supplier_name = spec.get('name')

# All other keys must be parsed out by this point.

if isinstance(params, dict):
	result = Supplier.get_supplier_by_name(name)(
		name=supplier_name, kurfile=kurfile, **params)
elif isinstance(params, str):
	result = Supplier.get_supplier_by_name(name)(params,
		name=supplier_name, kurfile=kurfile)
elif isinstance(params, (list, tuple)):
	result = Supplier.get_supplier_by_name(name)(*params,
		name=supplier_name, kurfile=kurfile)

#### Must dive into MnistSupplier(name=supplier_name, kurfile=kurfile, **params) ####
		"""

		logger.info("\n\nGet detailed info on data provider in kurfile: params\n\nget the name of Supplier class: name = candidates.pop() \n\nGet the Supplier class and instantiate it\n\n%s\n\nFinally, return the data supplier objects\n\n", create_supplier_obj)

		name = candidates.pop()
		params = spec[name]

		supplier_name = spec.get('name')

		# All other keys must be parsed out by this point.

		if isinstance(params, dict):
			result = Supplier.get_supplier_by_name(name)(
				name=supplier_name, kurfile=kurfile, **params)
		elif isinstance(params, str):
			result = Supplier.get_supplier_by_name(name)(params,
				name=supplier_name, kurfile=kurfile)
		elif isinstance(params, (list, tuple)):
			result = Supplier.get_supplier_by_name(name)(*params,
				name=supplier_name, kurfile=kurfile)
		else:
			raise ValueError('Expected the Supplier to be given a dictionary, '
				'list, tuple, or string for parameters. Instead, we received: '
				'{}'.format(params))

		logger.info("\n\nFirst, see supplier class, specified params on data supplier; \n\nthen see inside data supplier object\n\n")
		if logger.isEnabledFor(logging.INFO):
			print("Supplier class: {}\n".format(Supplier.get_supplier_by_name(name)))
			pprint(params)
			print("\n")
			pprint(result.__dict__)
			print("\n\n")
		return result

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.

			# Return value

			A lower-case string unique to this supplier.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_suppliers():
		""" Returns an iterator to the names of all suppliers.
		"""
		for cls in get_subclasses(Supplier):
			yield cls

	###########################################################################
	@staticmethod
	def get_supplier_by_name(name):
		""" Finds a supplier class with the given name.
		"""
		name = name.lower()
		for cls in Supplier.get_all_suppliers():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such supplier with name "{}"'.format(name))

	###########################################################################
	@classmethod
	def merge_suppliers(cls, suppliers):
		""" Merges a number of suppliers together, usually in order to create
			a data Provider.

			# Arguments

			suppliers: list of Supplier instance. The Suppliers to merge.

			# Return value

			A dictionary whose keys are the names of data sources and whose
			values are sources corresponding to those keys.
		"""
		result = {}
		for supplier in suppliers:
			sources = supplier.get_sources()
			for k, v in sources.items():
				if k not in result:
					# If we haven't added it before, do so now.
					result[k] = v
				elif v.is_derived():
					# We don't need to stack derived outputs.
					pass
				elif isinstance(result[k], StackSource):
					# Add this to the stack.
					logger.trace('Stacking data source: %s', k)
					result[k].stack(v)
				else:
					# Create a stack.
					logger.trace('Stacking data source: %s', k)
					result[k] = StackSource(result[k], v)
		return result

	###########################################################################
	def __init__(self, name=None, kurfile=None):
		""" Creates a new supplier.
		"""
		self.name = name
		self.kurfile = kurfile

	###########################################################################
	def is_delegate(self):						# pylint: disable=no-self-use
		""" Returns True if this supplier supplies suppliers.
		"""
		return False

	###########################################################################
	def get_delegates(self):					# pylint: disable=no-self-use
		""" Returns a dictionary or list of suppliers that this supplier
			supplies.

			This is important for nesting suppliers, for applications such as
			splitting.

			# Notes

			- If the Supplier implementation doesn't have any sub-suppliers,
			  then this call should return an empty dictioanry or empty list.
			- If the Supplier implementation does have sub-suppliers, then the
			  `is_delegate()` should return True, but calling `get_sources()`
			  should raise an exception.
		"""
		return {}

	###########################################################################
	def get_sources(self, sources=None):
		""" Returns a dictionary or list of data sources, as provided by this
			supplier.

			# Arguments

			sources: int or str, or a list of int or str. The sources to
				return. Passing a bare integer or string is equivalent to
				passing a list containing that single item. Only sources with
				those indices (or keys) are returned. If sources is None, then
				all sources are returned.

			# Return value

			A dictionary or list of Source instances.
		"""
		raise NotImplementedError

	###########################################################################
	def get_source(self, source):
		""" Returns a single data source.

			# Arguments

			source: str or int. If this supplier produces a list of sources,
				then this is the index of the source (as an integer). If this
				supplier produces a dictionary of sources, then this is the key
				corresponding to the desired source (as a string).

			# Return value

			A Source instance.
		"""
		# By default, we'll just use our more general `get_sources()` call to
		# do all the work.
		return get_any_value(self.get_sources(sources=(source, )))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

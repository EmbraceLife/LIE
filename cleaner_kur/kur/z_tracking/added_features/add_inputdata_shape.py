

"""
# How to know the input dataset shapes?
- kur data | build | train don't provide direct info on dataset shapes
- so added this feature to display dataset shapes when `kur data`
- Inset the following code inside `__main__.prepare_data()`
- Right above print data samples code

"""
# added_features: print out dataset shapes
		for k, v in batch.items():
			print("See a batch's keys and shapes: \n")
			print("key:", k)
			print("shape:", v.shape, "\n")

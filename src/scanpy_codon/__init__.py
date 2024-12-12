# from python import anndata.AnnData, anndata.concat

# Python likes "from . import plotting as pl"
# But codon hates it
import plotting as pl
import preprocessing as pp
import tools as tl

import get
import logging

# from .readwrite import read, read_10x_h5, read_10x_mtx, read_visium, write
from .readwrite import read_10x_h5
# For some reason, codon does not export anything not directly defined in this file
# In order to benchmark this I have made a 'ladder' to call functions
def read_10x_h5(
	filename: str,
):
	from .readwrite import read_10x_h5
	print("Ladder 1")
	return read_10x_h5(filename)

def pca_ladder(adata):
	import preprocessing as pp
	return pp.pca(adata)

def add_these(a: int, b: int):
	return a + b

# Seems unable to export globals, deploying stupid solution
@python
def get_settings():
	# https://stackoverflow.com/questions/24946321/write-a-no-op-or-dummy-class-in-python
	class DummySettings:
		def __init__(*args, **kwargs):
			pass
		def __call__(self, *args, **kwargs):
			return self
		def set_figure_params(self, *args, **kwargs):
			return self
		def __getattr__(self, *args, **kwargs):
			return self
	print("returning dummy settings")
	return DummySettings()



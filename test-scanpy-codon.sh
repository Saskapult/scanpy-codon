source ../.venv/bin/activate
export CODON_PYTHON="/usr/lib/libpython3.12.so"

python3 setup.py build_ext --inplace

mv scanpy_codon.cpython-312-x86_64-linux-gnu.so ./tests_demo/
cd ./tests_demo/
python3 preprocessing_clustering.py

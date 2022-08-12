# async-zarr

Hack wrapper around zarr to make data access async

To run normal example:

```bash
jupyter notebook ./example/notebook.ipynb
```

To run pyscript example:

```bash
pip install -e .
python ./azarr/tests/conftest.py
open http://localhost:8000/example/index.html  # browse to this location
```

And copy the cells from the notebook into the REPL cells.

# How to create virtual environment

- Find python.exe location and run terminal from there

- Use command 
```shell
python -m venv "<YOUR DESIRED VENV LOCATION>\<VENV NAME>"
```

- Then open your created venv 
```shell
cd "<YOUR DESIRED VENV LOCATION>\<VENV NAME>\Scripts"
```

- Activate it, when it is activated you must see "(<VENV NAME>)" on the left of your shell
```shell
activate
```

## Then you can install every package you want inside your created venv

- You can simply check which executable python you currently use, if it shows you path to your created venv you are good
```shell
python -c "import sys; print(sys.executable)"
```

- Then you can create IPython kernel, to use your venv in jupyter
```shell
python -m pip install -U pip
python -m pip install -U setuptools
pip install ipykernel
python -m ipykernel install --user --name <NAME OF THE KERNEL>
```
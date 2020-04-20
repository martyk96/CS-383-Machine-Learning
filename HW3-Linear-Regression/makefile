venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p python3 venv
	venv/bin/pip install -Ur requirements.txt
	touch venv/bin/activate

devbuild: venv
	venv/bin/python setup.py install

devinstall: venv
	venv/bin/pip install --editable .

problem1: venv
	venv/bin/python3 problem1.py
problem2: venv
	venv/bin/python3 problem2.py
problem3: venv
	venv/bin/python3 problem3.py $(datafile)
problem4: venv
	venv/bin/python3 problem4.py $(s) $(datafile)

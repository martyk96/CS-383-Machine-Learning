venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p python3 venv
	venv/bin/pip install -Ur requirements.txt
	touch venv/bin/activate

devbuild: venv
	venv/bin/python setup.py install

devinstall: venv
	venv/bin/pip install --editable .

test_bayes: venv
	venv/bin/python3 test.py bayes $(datafile)
test_logReg: venv
	venv/bin/python3 test.py logReg $(datafile)

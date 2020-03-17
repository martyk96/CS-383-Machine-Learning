venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p python3 venv
	venv/bin/pip install -Ur requirements.txt
	touch venv/bin/activate

test: venv
	venv/bin/python3 myKMeans.py

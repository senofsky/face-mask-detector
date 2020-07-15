# This Makefile handles the following operations for face_mask_detector:
#
#    - File formatting
#    - Static type checking
#    - Testing
#    - Packaging

FORMATTER=black -l 90 face_mask_detector test

# Check for correct formatting and perform type checking
.PHONY: check
check: check-format check-types

# Check for correct formatting
.PHONY: check-format
check-format:
	$(FORMATTER) --check

# Perform static type checking
.PHONY: check-types
check-types:
	mypy --strict face_mask_detector

# Reformat files
.PHONY: reformat
reformat:
	$(FORMATTER)

# Execute tests
.PHONY: test
test:
	make -C test all

test-%:
	make -C test $*

# Prepare a release by generating distribution packages
.PHONY: prep-release
prep-release: clean
	python setup.py sdist bdist_wheel
	twine check dist/*

# Upload the distribution packages to TestPyPi
.PHONY: upload-test-release
upload-test-release: prep-release
	twine upload --repository testpypi dist/*

# Upload the distribution packages to PyPi
.PHONY: upload-release
upload-release: prep-release
	twine upload dist/*

.PHONY: clean
clean:
	rm -rf sdist dist *.egg-info

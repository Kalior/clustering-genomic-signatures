SETUP_FILE = setup.py
DIRECTORIES = clustering distance vlmc

all:
	python3.6 $(SETUP_FILE) build_ext --inplace

clean:
	$(foreach dir, $(DIRECTORIES), rm -rf $(dir)/*.so $(dir)/*.c $(dir)/__pycache__)
	rm -rf build

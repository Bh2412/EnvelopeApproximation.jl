# Define directories
RUST_SRC_DIR = $(CURDIR)/rust/src
JULIA_LIB_DIR = $(CURDIR)
LIB_NAME = ring_dome_complement_intersection.so

# Define the target shared library path
TARGET = $(JULIA_LIB_DIR)/src/GeometricStressEnergyTensor/$(LIB_NAME)

# Default target
all: test

# Build the Rust library
$(TARGET): $(RUST_SRC_DIR)/lib.rs
	@echo "Building Rust library..."
	rustc $(RUST_SRC_DIR)/lib.rs --crate-type=cdylib -O -o $(TARGET)
	strip $(TARGET)
	@echo "Successfully built $(TARGET)"

# Clean built artifacts
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	@echo "Clean complete"

# Install target for potential integration with package managers
install: $(TARGET)
	@echo "Rust library already built at $(TARGET)"

# Test Rust code (add when you have Rust tests)
test-rust:
	cd rust && cargo test

# Test Julia code using Pkg.test()
test-julia:
	julia -e "using Pkg; Pkg.activate(\"$(JULIA_LIB_DIR)\"); Pkg.test()"

# Run all tests
test: $(TARGET) test-rust test-julia
	@echo "All tests completed"

.PHONY: all clean install test-rust test-julia test
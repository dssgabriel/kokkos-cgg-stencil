SRC := stencil.cxx
BIN := stencil
CXXFLAGS := -std=c++17 -Wall -Wextra -march=native -Ofast -finline-functions -funroll-loops -ftree-vectorize -fno-omit-frame-pointer -fopenmp -g

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $? -o $(BIN)

clean:
	@rm -rf $(BIN)

SRC:=$(wildcard *.cpp)
OBJ:=$(SRC:.cpp=.o)
EXEC:=dip3
CFLAGS:=-ggdb -O3
CXXFLAGS:=$(CFLAGS) $(shell pkg-config opencv --cflags)
LDFLAGS:=$(shell pkg-config opencv --libs --cflags)
LIBS:=

CC:=gcc
CXX:=g++

$(EXEC):$(OBJ)
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)
	
clean:
	rm -rf $(OBJ)
	
%.o:%.cpp
	$(CXX) -c $^ $(CXXFLAGS)
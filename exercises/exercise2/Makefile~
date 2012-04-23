SRC:=$(wildcard *.cpp)
OBJ:=$(SRC:.cpp=.o)

CFLAGS:=
CXXFLAGS:=-ggdb $(shell pkg-config opencv --cflags)
LDFLAGS:=$(shell pkg-config opencv --libs --cflags)

CC:=gcc
CXX:=g++

EXEC:=dip1

$(EXEC):$(OBJ)
	$(CXX) -o $@  $< $(LDFLAGS)
	
clean:
	-rm -rf $(OBJ)
	-rm -rf $(EXEC)
	
%.o:%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^

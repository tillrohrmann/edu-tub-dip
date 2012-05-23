SRC:=$(wildcard *.cpp)
OBJ:=$(SRC:.cpp=.o)
EXEC:=dip4
CXX:=g++

CFLAGS:= -ggdb -O3 $(shell pkg-config opencv --cflags)
CXXFLAGS:=$(CFLAGS)
LDFLAGS:= $(shell pkg-config opencv --libs)

$(EXEC):$(OBJ)
	$(CXX) -o $@ $< $(LDFLAGS) $(LIBS)
	
%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $^
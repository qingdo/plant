CC      = g++
CPPFLAGS= -O3 -c -std=c++11 -fopenmp  -march=native -w
LDFLAGS = -fopenmp -m64 -lpthread
SOURCES = hhl.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) hhl

hhl : $(OBJECTS)

	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o :
	$(CC) $(CPPFLAGS) $< -o $@

clean:
	rm -f *.o ../src/*.o hhl dump*

# Makefile for development
CC = gcc
CFLAGS = -Wall -Wextra -g -O0 -Iinclude
LDFLAGS = -lm

# Source files
LIB_SRC = src/nn.c
EXAMPLE_SRC = examples/xor_example.c

# Targets
.PHONY: all clean test

all: xor_example

xor_example: $(LIB_SRC) $(EXAMPLE_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: xor_example
	./xor_example

clean:
	rm -f xor_example

# Quick rebuild and test
quick: clean all test
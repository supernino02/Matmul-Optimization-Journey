#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct ArgParser
{
    int argc;
    char **argv;
    bool *visited; 
} ArgParser;

ArgParser *ArgParser_create(int argc, char **argv);

//destroy and check for unused arguments
void ArgParser_destroy(ArgParser *parser);

bool ArgParser_getFlag(ArgParser *parser, const char *key);
int ArgParser_getInt(ArgParser *parser, const char *key, int default_val);
double ArgParser_getDouble(ArgParser *parser, const char *key, double default_val);

#endif

#include "arg_parser.h"

ArgParser *ArgParser_create(int argc, char **argv)
{
    ArgParser *parser = (ArgParser *)malloc(sizeof(ArgParser));
    if (!parser)
    {
        perror("ArgParser allocation failed\n");
        exit(EXIT_FAILURE);
    }

    parser->argc = argc;
    parser->argv = argv;

    // Allocate bitmap (initialized to 0/false via calloc)
    parser->visited = (bool *)calloc(argc, sizeof(bool));
    if (!parser->visited)
    {
        perror("ArgParser bitmap allocation failed\n");
        free(parser);
        exit(EXIT_FAILURE);
    }

    // Always mark argv[0] (program name) as visited so it doesn't show up as unused
    if (argc > 0)
        parser->visited[0] = true;

    return parser;
}

void ArgParser_destroy(ArgParser *parser)
{
    //search for unused arguments 
    bool has_unused = false;
    for (int i = 0; i < parser->argc; i++)
        if (!parser->visited[i])
        {
            has_unused = true;
            break;
        }

    if (has_unused)
    {
        fprintf(stderr, "ERROR: invalid command-line arguments detected:\n");

        for (int i = 0; i < parser->argc; i++)
            if (!parser->visited[i])
                fprintf(stderr, "%s ", parser->argv[i]);
        fprintf(stderr, "\n");

        exit(EXIT_FAILURE);
    }

    //just clean up
    free(parser->visited);
    free(parser);
}

bool ArgParser_getFlag(ArgParser *parser, const char *key)
{
    for (int i = 0; i < parser->argc; i++)
    {
        if (strcmp(parser->argv[i], key) == 0)
        {
            parser->visited[i] = true; // Mark flag as used
            return true;
        }
    }
    return false;
}

int ArgParser_getInt(ArgParser *parser, const char *key, int default_val)
{
    for (int i = 0; i < parser->argc - 1; i++)
    {
        if (strcmp(parser->argv[i], key) == 0)
        {
            parser->visited[i] = true;     // Mark key
            parser->visited[i + 1] = true; // Mark value
            return atoi(parser->argv[i + 1]);
        }
    }
    return default_val;
}

double ArgParser_getDouble(ArgParser *parser, const char *key, double default_val)
{
    for (int i = 0; i < parser->argc - 1; i++)
    {
        if (strcmp(parser->argv[i], key) == 0)
        {
            parser->visited[i] = true;     // Mark key
            parser->visited[i + 1] = true; // Mark value
            return atof(parser->argv[i + 1]);
        }
    }
    return default_val;
}

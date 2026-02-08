#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csv_loader.h"
#include "matrix.h"

#define MAX_LINE 1024

// Count rows in CSV
static int count_lines(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
        return -1;

    int lines = 0;
    char buffer[MAX_LINE];
    while (fgets(buffer, MAX_LINE, fp))
    {
        if (strlen(buffer) > 1)
            lines++;
    }

    fclose(fp);
    return lines;
}

// Count columns (from first line)
static int count_cols(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
        return -1;

    char buffer[MAX_LINE];
    if (!fgets(buffer, MAX_LINE, fp))
    {
        fclose(fp);
        return -1;
    }

    int cols = 0;
    char *token = strtok(buffer, ",");
    while (token)
    {
        cols++;
        token = strtok(NULL, ",");
    }

    fclose(fp);
    return cols;
}

// Load CSV
int load_csv_to_matrix(const char *filename, Matrix *X, Matrix *y)
{
    int rows = count_lines(filename);
    int cols = count_cols(filename);
    if (rows <= 0 || cols <= 1) // need at least 1 feature + target
        return -1;

    *X = mat_create(rows, cols - 1);
    *y = mat_create(rows, 1);

    FILE *fp = fopen(filename, "r");
    if (!fp)
        return -1;

    char buffer[MAX_LINE];
    int i = 0;
    while (fgets(buffer, MAX_LINE, fp) && i < rows)
    {
        // remove newline
        buffer[strcspn(buffer, "\r\n")] = 0;

        char *token = strtok(buffer, ",");
        int j = 0;
        while (token && j < cols)
        {
            float val = atof(token);
            if (j < cols - 1)
            {
                mat_set(X, i, j, val);
            }
            else
            {
                mat_set(y, i, 0, val);
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(fp);
    return 0;
}

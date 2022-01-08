/*
 * Copyright (c) 2015, Nicolas Limare <nicolas@limare.net>
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under, at your option, the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version, or
 * the terms of the simplified BSD license.
 *
 * You should have received a copy of these licenses along this
 * program. If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @file xutils.inc.c
 * @brief misc utilities, mostly safe wrappers
 *
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 */

#ifndef _XUTILS_INC_C
#define _XUTILS_INC_C

#ifndef NOMEMALIGN
#define MEMALIGN 16
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L // posix_memalign()
#endif
#endif

#include <stdlib.h>
#include <stdio.h>

/** aligned memory allocation for SIMD */

/**
 * logging backend
 */
#define LOG(level, ...) {					\
	fprintf(stderr, level);					\
	fprintf(stderr, " ");					\
	fprintf(stderr, __VA_ARGS__);				\
	fprintf(stderr, " \t[from %s:%i]", __FILE__, __LINE__);	\
	fprintf(stderr, "\n");					\
    }

/**
 * error message macro
 */
#define ERROR(...) {			\
	LOG("ERROR", __VA_ARGS__);	\
	exit(EXIT_FAILURE);		\
    }

/**
 * info message macro
 */
#define INFO(...) {			\
	LOG("INFO ", __VA_ARGS__);	\
    }

/**
 * debug message macro
 */
#ifdef NDEBUG
#define DEBUG(...) {			\
	LOG("DEBUG", __VA_ARGS__);	\
    }
#else
#define DEBUG(...) {}
#endif

/**
 * NULL-safe malloc() wrapper
 */
static void* xmalloc(size_t size)
{
    if (0 == size)
        ERROR("zero-size allocation");
    void * ptr;
#ifndef NOMEMALIGN
    int success = posix_memalign(&ptr, MEMALIGN, size);
    if (0 != success)
        ERROR("memory allocation failed");
#else
    ptr = malloc(size);
    if (NULL == ptr)
        ERROR("memory allocation failed");
#endif
    return ptr;
}

/**
 * NULL-safe realloc() wrapper
 */
static void *xrealloc(void * ptr, size_t size)
{
    void * new_ptr = realloc(ptr, size);
    if (NULL == new_ptr)
        ERROR("out of memory");
    return new_ptr;
}

/**
 * NULL-aware free() wrapper
 */
static void xfree(void * ptr)
{
    if (NULL == ptr)
        ERROR("attempting to free a NULL pointer");
    free(ptr);
}

#endif                          /* !_XUTILS_INC_C */

# The TAFFO Annotation Language

The TAFFO Annotation Language is used to specify various properties on the
variables of the program, in order to allow TAFFO to properly infer the ranges
and the errors of all the other values involved in a computation.

The main components of the language are *attributes* and *data type
patterns*. Attributes are used to state a property of the variable being
annotated. Data type patterns are used to bind attributes to a specific part
of the variable; thus, most attributes are contained inside a data type
pattern.

**Note:**
Otherwise when explicitly specified, the syntax of the TAFFO annotation language is always case-sensitive.

**Note**
When specifying the syntax of an element of the language, multiple alternatives are
shown on separate lines. The actual syntax of the language ignores newlines.

## Literals

The TAFFO Annotation Language supports string and numeric literals.

### String literals

String literals are represented by a sequence of characters enclosed by
single quotes (`'`). Any quoted character except for `@` and `'` is interpreted
literally. The special sequence `@'` allows to include the single quote
character inside the string, and the special sequence `@@` allows to include
the at-sign character.

In this document, when the syntax expects a string literal, the word `<string>`
will appear instead.

### Integer-number literals

Integer-number literals are a sequence of one or more consecutive numeric
characters.
Such literals follow the same syntax as integer literals in the C programming
language. Thus, the `0` prefix indicates that the following characters
are a octal-base literal, and the `0x` prefix indicates that the following
characters are a hexadecimal-base literal.

In this document, when the syntax expects an integer literal, the word `<int>`
will appear instead.

### Boolean literals

Boolean literals are words that specify a binary truth value. The words `true`
and `yes` are used to specify a true value. The words `false` and `no` are used
to specify a false value. It is not possible to implicitly cast an integer
literal to a boolean literal.

In this document, when the syntax expects a boolean literal, the word `<bool>`
will appear instead.

### Real-number literals

Real-number literals are similar to integer-number literals but also have
a decimal part. The decimal separator is the dot (`.`). Again, the syntax is
the same as the one used by the C programming language.

In this document, when the syntax expects a real-number literal, the word
`<real>` will appear instead.

## Top-Level Syntax

At the topmost level, an annotation string is composed by zero or more top-level
attributes, and exactly one data type pattern except for `void`. The order of these
elements are irrelevant, and they are optionally separated by any amount of whitespace.

Example:

```
target('accum') backtracking scalar()
```

## Top-Level Attributes

A top-level attribute can appear only once, and is not enclosed in any
data type pattern. Such attributes are used to enable specific features of
TAFFO which do not depend on the data type of the variable.

Note that when multiple such attributes of the same type are present, only the last
ones of each type are evaluated.

### *errtarget*

```
errtarget(<string>)
```

The `errtarget` attribute sets the specified string as the name to use to
display the error computed by the Error Propagation TAFFO pass.

### *target*

```
target(<string>)
```

The `target` attribute does the same as `errtarget`,
i.e., it sets the specified string as the name to use to
display the error computed by the Error Propagation TAFFO pass.
Additionally, it also sets the function containing the annotated variable
as a _starting-point_ (see the related section at the end of this page).

### *backtracking*

```
backtracking(<int>)
backtracking(<bool>)
backtracking
```

The `backtracking` attribute, specifies if Initialization should schedule for conversion the values indirectly used by this variable, in addition to the values which use it.
The integer parameter to the attribute specifies the maximum recursive depth of the backtracking traversal -- in other words, the maximum DFG node distance between two instructions affected by it. Specifying the literal `0` disables backtracking.

The `backtracking(<bool>)` syntax allows to enable unlimited-depth backtracking by specifying the parameter `true`. `backtracking(false)` and `backtracking(0)` are synonyms.

The `backtracking` syntax is equivalent to writing `backtracking(yes)`.

## Data Type Patterns

Data type patterns contain the attributes that relate to the actual data stored
in the variable. We refer to such attributes as *data attributes*.

In this document, when the syntax expects a data type pattern inside another data type pattern, the word `<data type pattern>` will appear instead. Likewise, the same holds
for data attributes, and the word which will appear instead is `<data attribute>`.

When one or more of the same syntactic element can appear, three dots (`...`) will appear
in the syntax after the syntactic element that can be repeated.

### *scalar*

```
scalar()
scalar(<data attribute> ...)
```

The `scalar` data type pattern is used for homogeneous data types, like variables of
primitive types, or even arrays and matrices (despite what the `scalar` keyword would
suggest).

### *struct*

```
struct[]
struct[<data type pattern>]
struct[<data type pattern>, ... <data type pattern>]
```

The `struct` data type pattern is used for eterogeneous data types, in other words
structures. It contains a list of data type patterns, each of them corresponding in order
to the types contained in the struct.

Note that the syntax uses square brackets to help keep track of recursive structures, and the use of commas to emphasize that each element refers to a different struct item. 

### *void*

```
void
```

The `void` data type pattern is not valid at the topmost level, and is used for those
struct items that shall not be converted, and thus have no information associated.

## Data Attributes

Data attributes are attributes used to specify properties of the data stored in the
variable.

### *range*

```
range(<real>, <real>)
```

The `range` attribute specifies the range of the values that can be stored in the
annotated variable. The first literal is the minimum value, the second literal is the
maximum value.

### *type*

```
type(<int> <int>)
type(signed <int> <int>)
type(unsigned <int> <int>)
```

The `type` attribute specifies the fixed point type that will be used by conversion
when converting the variables and its related instructions. This attribute exists mostly
for debugging purposes, and should not be used unless strictly necessary.

The two integer literals always refer to the total size in bits of the fixed point type, and
to the position of the fractional point in bits, respectively.
The presence of the `signed` keyword specifies that the type is signed, and the `unsigned`
keyword specifies an unsigned type.
The syntax without the `signed` and `unsigned` keywords is equivalent to the syntax with
the `signed` keyword.

### *error*

```
error(<real>)
```

The `error` keyword specifies the initial error that the Error Propagation pass assumes
for the variable.

### *disabled*

```
disabled
```

The `disabled` keyword specifies that this data item should not be modified by Conversion even though it is annotated.

### *final*

```
final
```

The `final` keyword prevents the value range analysis stage from widening the initial range supplied for this variable.
Note that this can be done as far as TAFFO is able to keep track of the annotated variable in LLVM IR, so this works best for arrays and structs.


## Starting-Point

Some of the modules contained in TAFFO (namely, the value-range and error analyses)
perform a static analysis of the input code by propagating data
(such as variable ranges) through the data-flow of the program,
roughly simulating its execution.
Thus, they need to know where the execution of the program normally starts from.
This is accomplished by marking one or more functions as _starting-points_.
TAFFO then performs its static analyses by assuming the program's execution
could start from any of such functions.
Typically, the `main` function is marked as a starting-point.

Note that at least one starting-point should always be specified
in a program, unless the `-propagate-all` command-line flag for
the value-range analysis is set.
Otherwise, TAFFO will not be able to infer value ranges correctly,
causing issues with the data-type allocation module.

There are two ways of defining the starting-points of a program.

### Through annotations

The `target` annotation has the effect of marking as a _starting-point_
the function containing the variable it refers to.
This annotation allows for defining multiple starting-point functions
in the same program.

### Through a special variable

If TAFFO finds a global variable named
```
__taffo_vra_starting_function
```
that is initialized with a pointer to a function,
it will mark that function as a _starting-point_.

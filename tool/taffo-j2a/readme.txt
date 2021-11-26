j2a is a library to load taffo anotations from a json file, and to serialize them to another json or to annotation syntax directly.

There are 2 main classes:
	AnnotationMap: is a map that can be queried for global, local, function or struct annotations. Each of this getters returns a string that is the taffo annotation associated to that particular function/global/local. Annotation map offers toJSON to serialize to the json sintax the whole content.
    AnnotationMaps can be built from a llvm::json::Array, so the default behaviour would be to parse a json file, and then provide the top level array to the StructAnnotation constructor.
	Once the constuctor has extracted the values from the array, the array can be deallocated.
	

	Annotation: a single annotation, that can contain all the currently used annotations.	
	An annotation can be inserted into a annotationmap or can be serialized directly to both json and taffo annotation, if you wish to serilize an annotation that refers to a struct a symbol table must be provided. For this reason it is best to insert an annotation in an annotationMap and let to the annotation map handle the struct references.



TAFFO-J2A:
	taffo j2a is a clang tool that allow the insertion of annotations in a c/c++ code.
	The file on which to operate it must be a positional argument.

	The json file must be provided either with the -j inline annotation that expects the annotation string to be provided on the command line (-j="[ ogg1, ogg2 ]"), or it must be provided the name of a file with the -f option (-f="ann.json"). If none of this is present the lookup will be done against the annotation.json file in the current directory.

	If the -i is provided then the main file will be overwritten, if else the file will be writte on stdout.

	Compilation arguments must be provided after the -- option (taffo-j2a file.c -- -Ifolder/). if the -- is not provided then the tool will try to read the compile_commands.json file to find the flags.




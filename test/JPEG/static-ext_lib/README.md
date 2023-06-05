A compact implementation of a "baseline" JPEG decoder in C++, optimized with TAFFO framework.

Code organization
-----------------

The only public function is `decode` (from `decode.h/cpp`) which accepts either a path to a JPEG file or an input stream (should be opened in the binary mode). It returns `Image` (from `image.h`) with dimensions and RGB values for each pixel.

There are two additional source files: `decoding_tree.h` implements a "Huffman tree" required for the decoding and `util.h` contains some utility template code.

How a JPEG decoder works
---------------------------

The principle of operation of a JPEG decoder and its implementation are approximately as follows:

1. A JPEG file consists of several sections separated by certain "markers". These sections are processed by corresponding functions in our code (called `read...`).
2. A unit of data in JPEG format is a 8x8 table with some values. In JPEG they are written in a zig-zag order for certain reasons. So here we can finally apply our skills to fill a 2d array in some peculiar manner :)
2. To decode a compressed data the following information is required:
   1. Meta data from SOF0 section: image dimensions and subsampling coefficients for different channels. The idea here is that for less important color channels we can reduce its resolution before encoding.
   2. Quantization tables from DQT section: these determine image quality and compression level and are used to quantize initial values. When decoding we multiply by these values.
   3. Decoding trees from DHT section: these are used to compress the data using an algorithm similar to Huffman coding. For decoding we don't need to know its specifics, we only need to construct trees which map a code to an initial symbol.
3. After all this information is obtained we can process the main data in `SOS` section. Applying decoding trees we will get a set of 8x8 tables which then should be multiplied by quantization tables and transformed to the original values by applying the Inverse Discrete Cosine Transform.
4. Then these tables are lay out to the image canvas considering subsampling coefficients.
5. Then we convert from YCbCr color space to the traditional RGB space to get the final result. Gray scale images are decoded fine as well.

There are plenty of details, but there is enough information on the web (including official standard document) to figure  it all out.


Building with TAFFO
--------
Run the following command in the `src` directory
`taffo decoder.cpp -disable-vra -ljpeg -lfftw3 -o decoder-opt -O3`

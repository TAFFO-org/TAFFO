#include "decoder.h"
#include <array>
#include <cmath>
#include <experimental/optional>
#include <fstream>
#include <fftw3.h>
#include "decoding_tree.h"
#include "util.h"
#include <algorithm>
#include <iostream>
#include <jpeglib.h>
#include <cstring>
#include <hdf5_hl.h>
#include "bmp.h"
#include <dirent.h>
#include "IDCT.h"



namespace {

    double TOTAL_TIME=0;

    const int kBlockSide = 8;
    const int kBlockSize = kBlockSide * kBlockSide;
    const int kMaxCodeLength = 16;


    typedef enum {
        kDc = 0, kAc = 1, kNumCoefficientKinds = 2
    } CoefficientKind;
    typedef enum {
        kY = 0, kCb = 1, kCr = 2
    } Components;
    const int kNumQuantizationTables = 2;
    const int kNumDecodingTrees = 2;


    using Byte = uint8_t;
    using IntTable = std::array<int, kBlockSize>;
//using DoubleTable = std::array<double, kBlockSize>;
//using DoubleTable = double[kBlockSize];
    using QTable = std::experimental::optional<IntTable>;
    using Tree = std::experimental::optional<DecodingTree<Byte>>;


    const Byte kMarkerStart = 0xFF;
    const Byte kMarkerSOI = 0xD8;
    const Byte kMarkerCOM = 0xFE;
    const Byte kMarkerAPPMin = 0xE0;
    const Byte kMarkerAPPMax = 0xEF;
    const Byte kMarkerSOF0 = 0xC0;
    const Byte kMarkerDQT = 0xDB;
    const Byte kMarkerDHT = 0xC4;
    const Byte kMarkerSOS = 0xDA;
    const Byte kMarkerEOI = 0xD9;


    void assure(bool condition, std::string message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }


    void readBytes(std::istream &stream, Byte *buffer, size_t n) {
        stream.read(reinterpret_cast<char *>(buffer), n);
        assure(stream.good(), "Error reading the stream.");
    }


    int concatenateBytes(Byte high, Byte low) {
        int ret = high;
        ret <<= 8;
        ret |= low;
        return ret;
    }


    struct SOF0Data {
        int nRows;
        int nCols;
        int nComponents;

        std::vector<int> V;
        std::vector<int> H;
        std::vector<int> qTableIds;

        int Vmax;
        int Hmax;
    };


    SOF0Data readSOF0(std::istream &stream) {
        SOF0Data ret;
        Byte buffer[4];

        readBytes(stream, buffer, 2);
        int length = concatenateBytes(buffer[0], buffer[1]);

        readBytes(stream, buffer, 1);
        assure(buffer[0] == 8, "Incorrect precision flag in SOF0.");

        readBytes(stream, buffer, 4);
        ret.nRows = concatenateBytes(buffer[0], buffer[1]);
        ret.nCols = concatenateBytes(buffer[2], buffer[3]);
        assure(ret.nRows > 0 && ret.nCols > 0, "One of the image dimensions is zero.");

        readBytes(stream, buffer, 1);
        int nComponents = buffer[0];
        int expectedLength = 8 + 3 * nComponents;
        assure(length == expectedLength, "Incorrect specified length for SOF0 section.");
        assure(nComponents == 1 || nComponents == 3, "Only 1 or 3 color components are supported.");

        for (int c = 0; c < nComponents; ++c) {
            readBytes(stream, buffer, 3);
            assure(buffer[0] == c + 1, "Incorrect component number in SOF0.");

            int H = buffer[1] >> 4;
            int V = buffer[1] & 0xF;
            assure((H == 1 || H == 2 || H == 4) && (V == 1 || V == 2 || V == 4),
                   "Incorrect sampling factor in SOF0.");

            ret.H.push_back(H);
            ret.V.push_back(V);

            assure(buffer[2] < kNumQuantizationTables, "Incorrect id for a quantization table.");
            ret.qTableIds.push_back(buffer[2]);
        }

        ret.Hmax = *std::max_element(ret.H.begin(), ret.H.end());
        ret.Vmax = *std::max_element(ret.V.begin(), ret.V.end());

        ret.nComponents = nComponents;

        return ret;
    }


    void skipSection(std::istream &stream) {
        Byte buffer[2];
        readBytes(stream, buffer, 2);
        int length = concatenateBytes(buffer[0], buffer[1]);
        assure(length >= 2, "Section length can't be less than 2.");
        std::vector<Byte> content(length - 2);
        readBytes(stream, content.data(), length - 2);
    }


    std::string readCOM(std::istream &stream) {
        Byte buffer[2];
        readBytes(stream, buffer, 2);
        int length = concatenateBytes(buffer[0], buffer[1]);
        length -= 2;
        assure(length >= 0, "Section length can't be less than 2.");
        std::vector<Byte> content(length);
        readBytes(stream, content.data(), length);
        return std::string(reinterpret_cast<char *>(content.data()), length);
    }


    void readDQT(std::istream &stream, std::vector<QTable> &tables) {
        Byte buffer[kBlockSize + 1];

        readBytes(stream, buffer, 2);
        int length = concatenateBytes(buffer[0], buffer[1]);
        int bytesLeft = length - 2;

        assure(bytesLeft > 0 && bytesLeft % (kBlockSize + 1) == 0, "Incorrect size of DQT section.");

        int nTables = bytesLeft / (kBlockSize + 1);
        for (int t = 0; t < nTables; ++t) {
            readBytes(stream, buffer, kBlockSize + 1);
            int precisionFlag = buffer[0] >> 4;
            if (precisionFlag != 0) {
                throw std::runtime_error("Incorrect precision flag for a quantization table.");
            }
            int id = buffer[0] & 0xF;
            assure(id < kNumQuantizationTables, "Incorrect id for a quantization table.");
            assure(!tables[id], "Duplicated id for a quantization table.");

            IntTable table;
            fillMatrixInZigzag(&buffer[1], table.begin(), kBlockSide, kBlockSide);
            tables[id] = table;
        }
    }


    void readDHT(std::istream &stream, std::vector<std::vector<Tree>> &trees) {
        Byte buffer[kMaxCodeLength + 1];

        readBytes(stream, buffer, 2);
        int length = concatenateBytes(buffer[0], buffer[1]);
        int bytesLeft = length - 2;

        while (bytesLeft > 0) {
            readBytes(stream, buffer, kMaxCodeLength + 1);
            bytesLeft -= kMaxCodeLength + 1;
            assure(bytesLeft >= 0, "Inconsistent length of DHT section.");

            int kind = buffer[0] >> 4;
            assure(kind < kNumCoefficientKinds, "Incorrect tree kind code in DHT.");

            int id = buffer[0] & 0xF;
            assure(id < kNumDecodingTrees, "Wrong id for a decoding tree.");

            auto &tree = trees[kind][id];
            assure(!tree, "Duplicated decoding tree id.");
            std::vector<size_t> counts(buffer + 1, buffer + kMaxCodeLength + 1);

            size_t codesTotal = std::accumulate(counts.begin(), counts.end(), 0);
            std::vector<Byte> codes(codesTotal);

            readBytes(stream, codes.data(), codesTotal);
            bytesLeft -= codesTotal;
            assure(bytesLeft >= 0, "Inconsistent length of DHT section.");

            tree.emplace(codes, counts);
        }
    }


    class TableDecoder {
    public:
        TableDecoder(std::istream &stream) : bitReader_(stream) {}

        void decode(DecodingTree<Byte> &dcTree, DecodingTree<Byte> &acTree, IntTable &table) {
            IntTable zigzagTable;
            std::fill(zigzagTable.begin(), zigzagTable.end(), 0);

            int position = 0;

            dcTree.reset();
            while (!dcTree.step(bitReader_.getNext() == 0 ? kLeft : kRight)) {}
            int nBits = dcTree.getValue();

            if (nBits == 0) {
                zigzagTable[position] = 0;
            } else {
                int value = bitReader_.getNext(nBits);
                zigzagTable[position] = invertIfNecessary(value, nBits);
            }
            ++position;

            acTree.reset();
            while (position < kBlockSize) {
                while (!acTree.step(bitReader_.getNext() == 0 ? kLeft : kRight)) {}

                int value = acTree.getValue();
                acTree.reset();

                if (value == 0) {
                    break;
                }

                int nZeros = value >> 4;
                int nBits = value & 0xF;

                if (nBits == 0) {
                    if (nZeros == 15) {
                        ++nZeros;
                    } else {
                        throw std::runtime_error("Incorrect SOS section.");
                    }
                }

                position += nZeros;
                assure(position < kBlockSize, "Incorrect SOS section.");

                if (nBits > 0) {
                    value = bitReader_.getNext(nBits);
                    zigzagTable[position] = invertIfNecessary(value, nBits);
                    ++position;
                }
            }

            fillMatrixInZigzag(zigzagTable.data(), table.begin(), kBlockSide, kBlockSide);
        }

    private:
        class BitReader {
        public:
            BitReader(std::istream &stream) : stream_(stream), shift_(0) {}

            int getNext() {
                if (shift_ == 0) {
                    byte_ = readByte();
                    if (byte_ == 0xFF) {
                        assure(readByte() == 0, "Unexpected marker in SOS section found.");
                    }
                    shift_ = 8;
                }
                --shift_;
                return (byte_ >> shift_) & 1;
            }

            int getNext(int n) {
                int ret = 0;
                for (int i = 0; i < n; ++i) {
                    int bit = getNext();
                    ret <<= 1;
                    ret += bit;
                }
                return ret;
            }

        private:
            std::istream &stream_;
            int shift_;
            int byte_;

            int readByte() {
                int byte = stream_.get();
                assure(stream_.good(), "Error reading the stream.");
                return byte;
            }
        };

        int invertIfNecessary(int value, int nBits) {
            if (value >> (nBits - 1) == 0) {
                value -= (1 << nBits) - 1;
            }
            return value;
        }

        BitReader bitReader_;
    };


    std::vector<std::vector<IntTable>> readSOS(std::istream &stream,
                                               std::vector<std::vector<Tree>> &trees,
                                               const SOF0Data &sof0Data) {
        Byte buffer[6];
        readBytes(stream, buffer, 3);
        int nComponents = buffer[2];
        assure(nComponents == sof0Data.nComponents, "Incorrect number of components in SOS.");

        int length = concatenateBytes(buffer[0], buffer[1]);
        assure(length == 6 + 2 * nComponents, "Incorrect length of SOS header specified.");

        readBytes(stream, buffer, 2 * nComponents);
        std::vector<int> dcIds(nComponents), acIds(nComponents);
        for (int c = 0; c < nComponents; ++c) {
            int id = buffer[2 * c];
            assure(id == c + 1, "Incorrect component id in SOS.");

            dcIds[c] = buffer[2 * c + 1] >> 4;
            acIds[c] = buffer[2 * c + 1] & 0xF;

            assure(dcIds[c] < kNumDecodingTrees && acIds[c] < kNumDecodingTrees,
                   "Incorrect id for a decoding tree in SOS.");
        }

        for (int c = 0; c < nComponents; ++c) {
            assure(trees[kAc][dcIds[c]] && trees[kDc][acIds[c]],
                   "Required decoding tree is not provided.");
        }

        // There are 3 useless bytes.
        readBytes(stream, buffer, 3);

        int nRowsMcu = kBlockSide * sof0Data.Vmax;
        int nColsMcu = kBlockSide * sof0Data.Hmax;
        int nMcu = ((sof0Data.nRows + nRowsMcu - 1) / nRowsMcu) *
                   ((sof0Data.nCols + nColsMcu - 1) / nColsMcu);

        std::vector<std::vector<IntTable>> tables(nComponents);
        TableDecoder tableDecoder(stream);
        for (int mcu = 0; mcu < nMcu; ++mcu) {
            for (int c = 0; c < nComponents; ++c) {
                std::vector<IntTable> &componentTables = tables[c];
                auto &dcTree = *trees[kDc][dcIds[c]];
                auto &acTree = *trees[kAc][acIds[c]];

                int nTables = sof0Data.H[c] * sof0Data.V[c];
                for (int t = 0; t < nTables; ++t) {
                    IntTable table;
                    tableDecoder.decode(dcTree, acTree, table);

                    if (!componentTables.empty()) {
                        table[0] += componentTables.back()[0];
                    }
                    componentTables.push_back(table);
                }
            }
        }

        return tables;
    }


    std::vector<int> layoutTables(const std::vector<double *> &tables, int nRows, int nCols,
                                  int V, int H, int Vmax, int Hmax) {
        int nRowsMCU = Vmax * kBlockSide;
        int nColsMCU = Hmax * kBlockSide;

        int rowDecimation = Vmax / V;
        int colDecimation = Hmax / H;

        std::vector<int> pixels(nRows * nCols);
        int rowMCU = 0;
        int colMCU = 0;

        int tableCounter;

        auto it = tables.begin();
        while (it != tables.end()) {
            for (int v = 0; v < V; ++v) {
                for (int h = 0; h < H; ++h) {
                    int rowStart = rowMCU + v * rowDecimation * kBlockSide;
                    int colStart = colMCU + h * colDecimation * kBlockSide;
                    int rowEnd = rowStart + rowDecimation * kBlockSide;
                    int colEnd = colStart + colDecimation * kBlockSide;

                    tableCounter = 0;
                    //auto value = it->begin();
                    for (int row = rowStart; row < rowEnd; row += rowDecimation) {
                        for (int col = colStart; col < colEnd; col += colDecimation) {
                            for (int i = 0; i < rowDecimation; ++i) {
                                for (int j = 0; j < colDecimation; ++j) {
                                    int r = row + i;
                                    int c = col + j;
                                    if (r < nRows && c < nCols) {
                                        //pixels[r * nCols + c] = static_cast<int>(std::round(*value));
                                        pixels[r * nCols + c] = static_cast<int>(std::round((*it)[tableCounter]));
                                    }
                                }
                            }
                            //++value;
                            ++tableCounter;
                        }
                    }
                    ++it;
                }
            }
            colMCU += nColsMCU;
            if (colMCU >= nCols) {
                colMCU = 0;
                rowMCU += nRowsMCU;
            }
        }

        return pixels;
    }


    double *applyIDCT(const IntTable &table) {
        //DoubleTable input, output;
        double *output;
        double input[kBlockSize];// __attribute((annotate("range -3000 3000")));// = static_cast<double *>(malloc(sizeof(double) * kBlockSize));
        output = static_cast<double *>(malloc(sizeof(double) * kBlockSize));

        //std::copy(table.begin(), table.end(), input);
        for(int i=0; i<table.size(); i++){
            input[i]=table[i];
        }


        int k = 0;
        #define fac  (0.5 / kBlockSide)
        #define fac0 1.41421356237309504880168872420969807856967187537694807317667973799
        for (int i = 0; i < kBlockSide; ++i) {
            for (int j = 0; j < kBlockSide; ++j) {
                input[k]=table[k];
                input[k] *= fac;
                if (i == 0) {
                    input[k] *= fac0;
                }
                if (j == 0) {
                    input[k] *= fac0;
                }
                ++k;
            }
        }


        double realInput[kBlockSize];
        for(int i=0; i<kBlockSize; i++){
            realInput[i]=input[i];
        }

        fftw_plan plan = fftw_plan_r2r_2d(kBlockSide, kBlockSide, realInput, output,
                                          FFTW_REDFT01, FFTW_REDFT01, 0);
        fftw_execute(plan);
        fftw_destroy_plan(plan);


        return output;

        //return idct2(table.data());
    }


    struct JpegData {
        SOF0Data sof0;
        std::vector<QTable> qTables;
        std::vector<std::vector<IntTable>> componentTables;
        std::string comment;
    };


    JpegData readJpeg(std::istream &stream) {
        Byte buffer[2];

        std::experimental::optional<SOF0Data> sof0;

        std::vector<std::vector<IntTable>> componentTables;
        std::vector<std::vector<Tree>> trees(kNumCoefficientKinds,
                                             std::vector<Tree>(kNumDecodingTrees));
        std::vector<QTable> qTables(kNumQuantizationTables);

        std::string comment;

        readBytes(stream, buffer, 2);
        if (buffer[0] != kMarkerStart || buffer[1] != kMarkerSOI) {
            throw std::runtime_error("SOI marker is not found at the beginning.");
        }

        bool eoi = false;
        while (!eoi) {
            readBytes(stream, buffer, 2);
            assure(buffer[0] == kMarkerStart, "JPEG marker is expected, but not found.");

            Byte marker = buffer[1];
            if (kMarkerAPPMin <= marker && marker <= kMarkerAPPMax) {
                skipSection(stream);
                continue;
            }

            switch (marker) {
                case kMarkerCOM:
                    comment = readCOM(stream);
                    break;
                case kMarkerSOF0: {
                    assure(!sof0, "Duplicate SOF0 section.");
                    sof0 = readSOF0(stream);
                    break;
                }
                case kMarkerDQT:
                    readDQT(stream, qTables);
                    break;
                case kMarkerDHT:
                    readDHT(stream, trees);
                    break;
                case kMarkerSOS: {
                    assure((bool) sof0, "SOS section found before SOF0.");
                    componentTables = readSOS(stream, trees, *sof0);
                    break;
                }
                case kMarkerEOI:
                    eoi = true;
                    break;
                default:
                    throw std::runtime_error("Unknown marker.");
            }
        }

        /*for (int i=0; i<trees.size(); i++){
            for (int j=0; j<trees[i].size(); j++){
                delete trees[i][j].value().;
            }
        }*/
        return {*sof0, qTables, componentTables, comment};
    }


}  // namespace


Image decode(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    assure((bool) stream, "Error opening the file.");




    Image image = decode(stream);




    stream.close();
    return image;
}


Image decode(std::istream &stream) {
    JpegData jpeg = readJpeg(stream);
    const SOF0Data &sof0 = jpeg.sof0;

    int nComponents = sof0.nComponents;
    std::vector<std::vector<double *>> colorTables(nComponents);

    clock_t t;
    t = clock();


    for (int c = 0; c < nComponents; ++c) {
        int tableId = sof0.qTableIds[c];
        assure((bool) jpeg.qTables[tableId], "Required quantization table wasn't defined.");
        IntTable &qTable = *jpeg.qTables[tableId];
        for (IntTable &table : jpeg.componentTables[c]) {
            for (int i = 0; i < kBlockSize; ++i) {
                table[i] *= qTable[i];
            }

            double *data = applyIDCT(table);

            colorTables[c].push_back(data);
        }
    }

    jpeg.qTables.clear();
    jpeg.componentTables.clear();

    int nRows = sof0.nRows;
    int nCols = sof0.nCols;

    std::vector<std::vector<int>> channels(nComponents);
    for (int c = 0; c < nComponents; ++c) {
        channels[c] = layoutTables(colorTables[c], nRows, nCols,
                                   sof0.V[c], sof0.H[c], sof0.Vmax, sof0.Hmax);
    }

    /*Free pre allocated vectors*/
    for (int c = 0; c < nComponents; ++c) {
        for (int j = 0; j < colorTables[c].size(); j++) {
            free(colorTables[c][j]);
        }

    }

    std::vector<RGB> pixels(nRows * nCols);




    double r __attribute((annotate("range -50 300")));
    double g __attribute((annotate("range -50 300")));
    double b __attribute((annotate("range -50 300")));
    for (int i = 0, k = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j, ++k) {
            int y = channels[kY][k];
            int cb = 0;
            int cr = 0;
            if (nComponents == 3) {
                cb = channels[kCb][k];
                cr = channels[kCr][k];
            }

            r = y + 1.402 * cr + 128;
            g = y - 0.34414 * cb - 0.71414 * cr + 128;
            b = y + 1.772 * cb + 128;

            r = (clip_rgb(r));
            g = (clip_rgb(g));
            b = (clip_rgb(b));

            //printf("Pixel: %f %f %f\n", r, g, b);
            pixels[k] = {(int) r, (int) g, (int) b};
            //printf("Pixel: %d %d %d\n", pixels[k].r, pixels[k].g, pixels[k].b);
        }
    }
    Image image = {nRows, nCols, std::move(pixels)};


    t = clock() - t;
    double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    TOTAL_TIME += time_taken;
    printf("Time taken: %f \n", time_taken);

    channels.clear();

    return image;
}


Image readJpg_using_library(const std::string &filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr err;
    FILE *infile = fopen(filename.c_str(), "rb");

    if (!infile) {
        throw std::runtime_error("Can't open " + filename + " for reading");
    }

    cinfo.err = jpeg_std_error(&err);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, (boolean) true);
    (void) jpeg_start_decompress(&cinfo);

    int row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo,
                                                   JPOOL_IMAGE, row_stride, 1);

    int nRows = cinfo.output_height;
    int nCols = cinfo.output_width;
    std::vector<RGB> pixels(nRows * nCols);
    size_t k = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
        for (size_t x = 0; x < nCols; ++x) {
            RGB &pixel = pixels[k];
            if (cinfo.output_components == 3) {
                pixel.r = buffer[0][x * 3];
                pixel.g = buffer[0][x * 3 + 1];
                pixel.b = buffer[0][x * 3 + 2];
            } else {
                pixel.r = pixel.g = pixel.b = buffer[0][x];
            }
            ++k;
        }
    }
    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return {nRows, nCols, std::move(pixels)};
}


int main(int argc, char *argv[]) {
    if (argc != 2){
        printf("Usage: %s path-containing-only-jpeg", argv[0]);
        return 0;
    }

    DIR *dir;
    struct dirent *ent;
    char* path = argv[1];
    //char path[] = "./jpeg_samples/";
    int processed=0;

    if ((dir = opendir (path)) != nullptr) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != nullptr) {
            if(ent->d_type == DT_REG){
                printf ("Decoding %s (%d/5000)...\n", ent->d_name, processed+1);


                std::string s(ent->d_name);

                std::string fileName = path + s;
                Image decoded = decode(fileName);
                //write_bmp_file((fileName + ".bmp").c_str(), decoded.nCols, decoded.nRows,  24, decoded.pixels);
                Image decode_standard_lib = readJpg_using_library(fileName);
                compare(decoded, decode_standard_lib);

                decoded.pixels.clear();
                decode_standard_lib.pixels.clear();

                fflush(stdout);

                //we are not taking into account the time of cleanup, please!
                fftw_cleanup();
                //printf("Maxval=%f\nMinval=%f\n\n", max_val, min_val);

                processed ++;
            }else{
                printf ("Not a regular file: %s\n", ent->d_name);
            }

        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("Cannot open given directory...");
        return EXIT_FAILURE;
    }

    printf("Total kernel time: %f", TOTAL_TIME);

    /*for (int i = 1; i <= testCount; ++i) {
        std::string fileName = "jpeg_samples/" + std::to_string(i) + ".jpg";
        auto decoded = decode(fileName);
        write_bmp_file((fileName + ".bmp").c_str(), decoded.nCols, decoded.nRows,  24, decoded.pixels);
        compare(decoded, readJpg_using_library(fileName));
    }*/


}


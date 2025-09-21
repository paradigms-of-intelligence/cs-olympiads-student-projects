#include "../include/jxl/encode.h"
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

/**
 * Loads an image from a text file into a float vector.
 *
 * Expected file format:
 *   - The first two numbers: width and height
 *   - Then pixel values (width * height * num_channels numbers).
 *
 * @param path Path to the input file.
 * @param num_channels Number of color channels (e.g., 3 for RGB).
 * @param bytes_per_pixel Number of bytes per pixel component.
 * @param width (output) Width of the image.
 * @param height (output) Height of the image.
 * @return A flat vector of floats containing the pixel values.
 */
vector<float> loadImage(const string &path, size_t num_channels, size_t &width,
                        size_t &height) {
    ifstream file(path);
    if (!file) {
        throw runtime_error("Failed to open file: " + path);
    }

    // Read image dimensions
    file >> width >> height;
    cout << "Image size (" << path << "): " << width << "x" << height << endl;

    // Allocate memory for pixel values
    const size_t frame_bytes = width * height * num_channels;
    vector<float> frame(frame_bytes);

    // Read pixel data in row-major order
    for (size_t i = 0; i < width * height; ++i) {
        for (size_t c = 0; c < num_channels; ++c) {
            file >> frame[i * num_channels + c];
        }
    }

    file.close();
    return frame;
}

int main(int argc, char *argv[]) {
    const size_t num_channels = 3;    // RGB
    const size_t bytes_per_pixel = 4; // 32-bit float per component

    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " <N> <image1> <image2> ... <imageN> <output> [d1 d2 "
                "... dN]"
             << endl;
        return 1;
    }

    // Number of input images
    int N = stoi(argv[1]);

    // TODO: add support for 3+ layer encoding
    if (N != 2) {
        cerr << "The encoder currently only supports 2 layers. Recieved " << N
             << " layers instead" << endl return 1;
    }

    if (argc < 3 + N) {
        cerr << "Not enough arguments for N=" << N << endl;
        return 1;
    }

    string output_path = argv[2 + N];
    bool has_distances =
        (argc >= 3 + N + N); // optional distances for each frame

    vector<size_t> widths(N), heights(N);
    vector<vector<float>> frames;

    // Load all images into memory
    for (int i = 0; i < N; i++) {
        size_t w, h;
        vector<float> f = loadImage(argv[2 + i], num_channels, w, h);
        widths[i] = w;
        heights[i] = h;
        frames.push_back(std::move(f));
    }

    // Create JPEG XL encoder
    JxlEncoder *enc = JxlEncoderCreate(nullptr);

    // Set basic image info (use the first image dimensions as reference)
    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = widths[0];
    basic_info.ysize = heights[0];
    basic_info.bits_per_sample = 8; // 8-bit per channel
    basic_info.uses_original_profile = JXL_FALSE;
    basic_info.have_animation = false;
    basic_info.animation.tps_numerator = 1;
    basic_info.animation.tps_denominator = 1;
    basic_info.animation.num_loops = 0;
    JxlEncoderSetBasicInfo(enc, &basic_info);

    // Set color encoding to sRGB
    JxlColorEncoding color_encoding;
    JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
    JxlEncoderSetColorEncoding(enc, &color_encoding);

    // Define pixel format: RGB float
    JxlPixelFormat pixel_format = {num_channels, JXL_TYPE_FLOAT,
                                   JXL_NATIVE_ENDIAN, 0};

    // Add all lower-resolution layers (frames 1..N-1)
    for (int i = 1; i < N; i++) {
        JxlEncoderFrameSettings *frame_settings =
            JxlEncoderFrameSettingsCreate(enc, nullptr);

        // Set resampling factor (power of 2 downscaling)
        int scale_factor = 1 << i;
        cout << "Adding frame " << i << " with scale factor " << scale_factor
             << endl;

        JxlEncoderFrameSettingsSetOption(
            frame_settings, JXL_ENC_FRAME_SETTING_RESAMPLING, scale_factor);
        JxlEncoderFrameSettingsSetOption(
            frame_settings, JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED, 1);

        // If distances are provided, use them
        if (has_distances) {
            double d = atof(argv[3 + N + i]);
            JxlEncoderSetFrameDistance(frame_settings, d);
        }

        // Add frame data to encoder
        JxlEncoderAddImageFrame(frame_settings, &pixel_format, frames[i].data(),
                                widths[i] * heights[i] * num_channels *
                                    bytes_per_pixel);
    }

    // Add the highest-resolution layer (first image)
    {
        JxlEncoderFrameSettings *frame_settings =
            JxlEncoderFrameSettingsCreate(enc, nullptr);

        // Initialize frame header
        JxlFrameHeader *frame_header = new JxlFrameHeader;
        JxlEncoderInitFrameHeader(frame_header);
        std::cout << "JxlFrameHeader initialized." << std::endl;

        // Optionally set distance if provided
        if (argv[4]) {
            JxlEncoderSetFrameDistance(frame_settings, atof(argv[4]));
        }

        // Apply header and add frame data
        JxlEncoderSetFrameHeader(frame_settings, frame_header);
        JxlEncoderAddImageFrame(frame_settings, &pixel_format, frames[0].data(),
                                widths[0] * heights[0] * num_channels *
                                    bytes_per_pixel);
    }

    // No more frames will be added
    JxlEncoderCloseInput(enc);

    // Allocate initial output buffer
    std::vector<uint8_t> compressed(1024);
    uint8_t *next_out = compressed.data();
    size_t avail_out = compressed.size();

    // Encode and expand buffer as needed
    JxlEncoderStatus status;
    do {
        status = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
        if (status == JXL_ENC_NEED_MORE_OUTPUT) {
            size_t offset = next_out - compressed.data();
            compressed.resize(compressed.size() * 2);
            next_out = compressed.data() + offset;
            avail_out = compressed.size() - offset;
        }
    } while (status == JXL_ENC_NEED_MORE_OUTPUT);

    if (status != JXL_ENC_SUCCESS) {
        std::cerr << "Encoding failed!" << std::endl;
        JxlEncoderDestroy(enc);
        return 1;
    }

    // Save compressed data to file
    size_t compressed_size = next_out - compressed.data();
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        JxlEncoderDestroy(enc);
        return 1;
    }

    out.write(reinterpret_cast<const char *>(compressed.data()),
              compressed_size);
    out.close();

    std::cout << "Saved " << output_path << " with size " << compressed_size
              << " bytes\n";

    // Clean up encoder
    JxlEncoderDestroy(enc);
    return 0;
}

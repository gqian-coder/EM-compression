// Compress the 4D STEM data
// Data oragnization: scatter_px, scatter_py, vx, vy

#include <stdio.h>
#include <string.h>  // For memcpy
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <chrono>
#include <utility>

#include "mgard/compress_x.hpp"

#if defined(MGARD_STEM_USE_HIP_EVENTS) && MGARD_STEM_USE_HIP_EVENTS
#include <hip/hip_runtime.h>
#endif

using namespace std;

namespace {

const char *to_string(mgard_x::compress_status_type s) {
    switch (s) {
    case mgard_x::compress_status_type::Success:
        return "Success";
    case mgard_x::compress_status_type::Failure:
        return "Failure";
    case mgard_x::compress_status_type::OutputTooLargeFailure:
        return "OutputTooLargeFailure";
    case mgard_x::compress_status_type::NotSupportHigherNumberOfDimensionsFailure:
        return "NotSupportHigherNumberOfDimensionsFailure";
    case mgard_x::compress_status_type::NotSupportDataTypeFailure:
        return "NotSupportDataTypeFailure";
    case mgard_x::compress_status_type::BackendNotAvailableFailure:
        return "BackendNotAvailableFailure";
    default:
        return "Unknown";
    }
}

struct CompressTimingStats {
        size_t calls = 0;
        double total_ms = 0.0;
        double min_ms = std::numeric_limits<double>::infinity();
        double max_ms = 0.0;

        void add(double ms) {
                ++calls;
                total_ms += ms;
                min_ms = std::min(min_ms, ms);
                max_ms = std::max(max_ms, ms);
        }
};

struct WallTimingStats {
    size_t calls = 0;
    double total_s = 0.0;
    double min_s = std::numeric_limits<double>::infinity();
    double max_s = 0.0;

    void add(double s) {
        ++calls;
        total_s += s;
        min_s = std::min(min_s, s);
        max_s = std::max(max_s, s);
    }
};

struct HostBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    bool hip_pinned = false;

    void reset() {
        if (!ptr) return;
#if defined(MGARD_STEM_USE_HIP_EVENTS) && MGARD_STEM_USE_HIP_EVENTS
        if (hip_pinned) {
            (void)hipHostFree(ptr);
        } else {
            std::free(ptr);
        }
#else
        std::free(ptr);
#endif
        ptr = nullptr;
        bytes = 0;
        hip_pinned = false;
    }

    ~HostBuffer() { reset(); }
};

#if defined(MGARD_STEM_USE_HIP_EVENTS) && MGARD_STEM_USE_HIP_EVENTS
class GpuTimer {
    public:
        GpuTimer() {
                (void)hipEventCreate(&start_);
                (void)hipEventCreate(&stop_);
        }

        ~GpuTimer() {
            (void)hipEventDestroy(start_);
            (void)hipEventDestroy(stop_);
        }

        void start() {
            // Ensure previous work on any stream is complete so timing is isolated.
            (void)hipDeviceSynchronize();
            (void)hipEventRecord(start_, nullptr);
        }

        double stop_ms() {
            // MGARD-X may launch kernels on internal streams; synchronize so the
            // elapsed time reflects all device work.
            (void)hipDeviceSynchronize();
            (void)hipEventRecord(stop_, nullptr);
            (void)hipEventSynchronize(stop_);
                float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, start_, stop_);
                return static_cast<double>(ms);
        }

    private:
        hipEvent_t start_{};
        hipEvent_t stop_{};
};
#else
class GpuTimer {
    public:
        void start() { t0_ = std::chrono::steady_clock::now(); }
        double stop_ms() {
                auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double, std::milli>(t1 - t0_).count();
        }

    private:
        std::chrono::steady_clock::time_point t0_;
};
#endif

class CpuTimer {
    public:
        void start() { t0_ = std::chrono::steady_clock::now(); }
        double stop_s() {
                auto t1 = std::chrono::steady_clock::now();
                return std::chrono::duration<double>(t1 - t0_).count();
        }

    private:
        std::chrono::steady_clock::time_point t0_;
};

template <typename... Args>
mgard_x::compress_status_type timed_mgard_compress(CompressTimingStats &stats,
                           Args &&...args) {
    GpuTimer timer;
    timer.start();
    auto status = mgard_x::compress(std::forward<Args>(args)...);
    stats.add(timer.stop_ms());
    return status;
}

template <typename... Args>
mgard_x::compress_status_type timed_mgard_decompress(CompressTimingStats &stats,
                             Args &&...args) {
    GpuTimer timer;
    timer.start();
    auto status = mgard_x::decompress(std::forward<Args>(args)...);
    stats.add(timer.stop_ms());
    return status;
}

}

int main(int argc, char** argv) {

    CpuTimer total_timer;
    total_timer.start();

    std::string filename     = argv[1];
    int px = std::stoi(argv[2]);
    int py = std::stoi(argv[3]); 
    float tol = std::stof(argv[4]);
    float s   = 0.0;
    int ndim  = std::stoi(argv[5]);

    size_t nData = size_t(256)*256*256*256;
    std::vector<float> stemData;
    std::vector<float> stemData_rct;
    double alloc_stem_s = 0.0;
    double alloc_rct_s  = 0.0;
    {
        CpuTimer t;
        t.start();
        std::vector<float> tmp(nData);
        stemData = std::move(tmp);
        alloc_stem_s = t.stop_s();
    }
    {
        CpuTimer t;
        t.start();
        std::vector<float> tmp(nData);
        stemData_rct = std::move(tmp);
        alloc_rct_s = t.stop_s();
    }

    double io_read_s = 0.0;
    {
        CpuTimer t;
        t.start();
        FILE *fp = fopen(filename.c_str(),"rb");
        fread(stemData.data(), sizeof(float), nData, fp);
        fclose(fp);
        io_read_s = t.stop_s();
    }

    size_t r_interval = 256;
    // double minmax_s = 0.0;
    // float minv = 0.0f;
    // float maxv = 0.0f;
    // {
    //     CpuTimer t;
    //     t.start();
    //     minv = *std::min_element(stemData.begin(), stemData.end());
    //     maxv = *std::max_element(stemData.begin(), stemData.end());
    //     minmax_s = t.stop_s();
    // }
    // tol = tol * (maxv - minv);
    // std::cout << "Data min = " << minv << ", max = " << maxv << ", tol = " << tol << "\n";
    // compression parameters
    mgard_x::Config config;
#if defined(MGARD_ENABLE_HIP)
    config.dev_type = mgard_x::device_type::HIP;
    std::cout << "Using HIP backend for MGARD-X\n";
#elif defined(MGARD_ENABLE_CUDA)
    config.dev_type = mgard_x::device_type::CUDA;
    std::cout << "Using CUDA backend for MGARD-X\n";
    
#else
    config.dev_type = mgard_x::device_type::SERIAL;
    std::cout << "Using SERIAL backend for MGARD-X\n";
#endif
    // config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    config.lossless = mgard_x::lossless_type::Huffman;
    config.normalize_coordinates = true;
    std::vector<mgard_x::SIZE> data_shape_2d{256, 256};
    std::vector<mgard_x::SIZE> data_shape_3d{256, 256, 256};
    std::vector<mgard_x::SIZE> data_shape_4d{r_interval, 256, 256, 256};
    std::vector<mgard_x::SIZE> data_shape_3d_4d{256, 256, 256*256};

    size_t dim2 = 256*256*256;
    size_t dim1 = 256*256;
   
    size_t bufferSize = (ndim==2) ? 256*256 : ((ndim==3) ? 256*256*256 : r_interval*256*256*256);
    bufferSize = bufferSize * sizeof(float);
    size_t compressed_size = bufferSize;
    size_t total_dataSize  = 0;
    double alloc_workbuf_s = 0.0;
    char *bufferOut = nullptr;
    void *compressed_data = nullptr;
    void *decompressed    = nullptr;
    {
        CpuTimer t;
        t.start();
        bufferOut = (char *) malloc(compressed_size);
        compressed_data = bufferOut;
        decompressed    = (void *) malloc(bufferSize);
        alloc_workbuf_s = t.stop_s();
    }
    size_t total_compressed = 0;
    CompressTimingStats compress_stats;
    CompressTimingStats decompress_stats;
    WallTimingStats compress_wall_stats;
    WallTimingStats decompress_wall_stats;
    WallTimingStats memcpy_wall_stats;

    // Optional pinned host buffers to reduce H2D/D2H overhead and make GPU timing
    // less sensitive to pageable-memory transfers.
    HostBuffer pinned_in;
    HostBuffer pinned_out;
    HostBuffer pinned_decompressed;
    WallTimingStats input_copy_wall_stats;
    WallTimingStats pinned_setup_wall_stats;

    auto try_alloc_pinned = [&](HostBuffer &buf, size_t bytes, const char *name) {
        CpuTimer t_total;
        t_total.start();
        buf.reset();
        buf.bytes = bytes;
        if (bytes == 0) {
            pinned_setup_wall_stats.add(t_total.stop_s());
            return;
        }

#if defined(MGARD_STEM_USE_HIP_EVENTS) && MGARD_STEM_USE_HIP_EVENTS
        // HIP pinned host allocation.
        void *p = nullptr;
        hipError_t err = hipHostMalloc(&p, bytes, hipHostMallocDefault);
        if (err == hipSuccess && p) {
            buf.ptr = p;
            buf.hip_pinned = true;
            std::cout << "Pinned host buffer allocated (HIP): " << name << ", bytes=" << bytes
                      << "\n";
            pinned_setup_wall_stats.add(t_total.stop_s());
            return;
        }
#endif

        // Fallback: allocate regular memory and attempt to pin/register via MGARD.
        buf.ptr = std::malloc(bytes);
        buf.hip_pinned = false;
        if (!buf.ptr) {
            std::cerr << "Warning: failed to allocate host buffer: " << name
                      << ", bytes=" << bytes << "\n";
            return;
        }

        mgard_x::pin_memory(buf.ptr, static_cast<mgard_x::SIZE>(bytes), config);
        const bool pinned = mgard_x::check_memory_pinned(buf.ptr, config);
        std::cout << "Host buffer allocated and registered: " << name << ", bytes=" << bytes
                  << ", pinned=" << (pinned ? "yes" : "no") << "\n";
        pinned_setup_wall_stats.add(t_total.stop_s());
    };

    // Allocate pinned buffers sized to the *per-call* working set.
    // If this fails (e.g., too large), we fall back to the original pointers.
    try_alloc_pinned(pinned_in, bufferSize, "compress_input");
    try_alloc_pinned(pinned_out, bufferSize, "compress_output");
    try_alloc_pinned(pinned_decompressed, bufferSize, "decompressed_output");

    if (pinned_out.ptr) {
        compressed_data = pinned_out.ptr;
    }
    if (pinned_decompressed.ptr) {
        std::free(decompressed);
        decompressed = pinned_decompressed.ptr;
    }
    if (ndim==2) { 
        if (px<0 && py<0) { // compress all data
            for (size_t r=0; r<256; r++) {
                std::cout << "r = " << r << "\n";
                size_t offset_r = r * dim2;
                for (size_t c=0; c<256; c++) {
                    if (c % 10==0) std::cout << "c = " << c << "\n";
                    size_t k = offset_r + c*dim1;
                    compressed_size = bufferSize; 
                    const void *input_ptr = &stemData.data()[k];
                    if (pinned_in.ptr) {
                        CpuTimer tcopy;
                        tcopy.start();
                        memcpy(pinned_in.ptr, input_ptr, bufferSize);
                        input_copy_wall_stats.add(tcopy.stop_s());
                        input_ptr = pinned_in.ptr;
                    }
                    CpuTimer wall;
                    wall.start();
                    auto status = timed_mgard_compress(
                        compress_stats, 2, mgard_x::data_type::Float, data_shape_2d, tol, s,
                        mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                        compressed_size, config, true);
                    compress_wall_stats.add(wall.stop_s());
                    if (status != mgard_x::compress_status_type::Success) {
                        std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                        return 2;
                    }
                    total_compressed += compressed_size;
                    total_dataSize   += bufferSize;
                    {
                        CpuTimer wall_d;
                        wall_d.start();
                        auto dstatus = timed_mgard_decompress(
                            decompress_stats, compressed_data, compressed_size, decompressed,
                            config, true);
                        decompress_wall_stats.add(wall_d.stop_s());
                        if (dstatus != mgard_x::compress_status_type::Success) {
                            std::cerr << "mgard_x::decompress failed: " << to_string(dstatus)
                                      << "\n";
                            return 3;
                        }
                    }
                    {
                        CpuTimer tmem;
                        tmem.start();
                        memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                        memcpy_wall_stats.add(tmem.stop_s());
                    }
                }
            }
        } else if (px>=0 and py>=0) { // compress only a 2D slice of data at r=px, c=py
            size_t offset_r = px * dim2;
            size_t k = offset_r + py*dim1;    
            compressed_size = bufferSize;
            const void *input_ptr = &stemData.data()[k];
            if (pinned_in.ptr) {
                CpuTimer tcopy;
                tcopy.start();
                memcpy(pinned_in.ptr, input_ptr, bufferSize);
                input_copy_wall_stats.add(tcopy.stop_s());
                input_ptr = pinned_in.ptr;
            }
            CpuTimer wall;
            wall.start();
            auto status = timed_mgard_compress(
                compress_stats, 2, mgard_x::data_type::Float, data_shape_2d, tol, s,
                mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                compressed_size, config, true);
            compress_wall_stats.add(wall.stop_s());
            if (status != mgard_x::compress_status_type::Success) {
                std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                return 2;
            }
            total_compressed += compressed_size;
            total_dataSize   += bufferSize;
            {
                CpuTimer wall_d;
                wall_d.start();
                auto dstatus = timed_mgard_decompress(
                    decompress_stats, compressed_data, compressed_size, decompressed, config,
                    true);
                decompress_wall_stats.add(wall_d.stop_s());
                if (dstatus != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::decompress failed: " << to_string(dstatus) << "\n";
                    return 3;
                }
            }
            {
                CpuTimer tmem;
                tmem.start();
                memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                memcpy_wall_stats.add(tmem.stop_s());
            }
        } else if (px >= 0) { // compress a 3D slice of data at r=px
            size_t offset_r = px * dim2; 
            for (size_t c=0; c<256; c++) {
                std::cout << "c = " << c << "\n";
                size_t k = offset_r + py*dim1;
                compressed_size = bufferSize;
                const void *input_ptr = &stemData.data()[k];
                if (pinned_in.ptr) {
                    CpuTimer tcopy;
                    tcopy.start();
                    memcpy(pinned_in.ptr, input_ptr, bufferSize);
                    input_copy_wall_stats.add(tcopy.stop_s());
                    input_ptr = pinned_in.ptr;
                }
                CpuTimer wall;
                wall.start();
                auto status = timed_mgard_compress(
                    compress_stats, 2, mgard_x::data_type::Float, data_shape_2d, tol, s,
                    mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                    compressed_size, config, true);
                compress_wall_stats.add(wall.stop_s());
                if (status != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                    return 2;
                }
                total_compressed += compressed_size;
                total_dataSize   += bufferSize;
                {
                    CpuTimer wall_d;
                    wall_d.start();
                    auto dstatus = timed_mgard_decompress(
                        decompress_stats, compressed_data, compressed_size, decompressed,
                        config, true);
                    decompress_wall_stats.add(wall_d.stop_s());
                    if (dstatus != mgard_x::compress_status_type::Success) {
                        std::cerr << "mgard_x::decompress failed: " << to_string(dstatus)
                                  << "\n";
                        return 3;
                    }
                }
                {
                    CpuTimer tmem;
                    tmem.start();
                    memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                    memcpy_wall_stats.add(tmem.stop_s());
                }
            }
        } else { // compress a 3D slice of data at c=py
            for (size_t r=0; r<256; r++) {
                std::cout << "r = " << r << "\n";
                size_t offset_r = r * dim2;
                size_t k = offset_r + py*dim1;
                compressed_size = bufferSize;
                const void *input_ptr = &stemData.data()[k];
                if (pinned_in.ptr) {
                    CpuTimer tcopy;
                    tcopy.start();
                    memcpy(pinned_in.ptr, input_ptr, bufferSize);
                    input_copy_wall_stats.add(tcopy.stop_s());
                    input_ptr = pinned_in.ptr;
                }
                CpuTimer wall;
                wall.start();
                auto status = timed_mgard_compress(
                    compress_stats, 2, mgard_x::data_type::Float, data_shape_2d, tol, s,
                    mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                    compressed_size, config, true);
                compress_wall_stats.add(wall.stop_s());
                if (status != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                    return 2;
                }
                total_compressed += compressed_size;
                total_dataSize   += bufferSize;
                {
                    CpuTimer wall_d;
                    wall_d.start();
                    auto dstatus = timed_mgard_decompress(
                        decompress_stats, compressed_data, compressed_size, decompressed,
                        config, true);
                    decompress_wall_stats.add(wall_d.stop_s());
                    if (dstatus != mgard_x::compress_status_type::Success) {
                        std::cerr << "mgard_x::decompress failed: " << to_string(dstatus)
                                  << "\n";
                        return 3;
                    }
                }
                {
                    CpuTimer tmem;
                    tmem.start();
                    memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                    memcpy_wall_stats.add(tmem.stop_s());
                }
            }
        }
    } else if (ndim==3) {
        if (px<0 && py<0) {
            for (size_t r=0; r<256; r++) {
                std::cout << "r = " << r << "\n";
                size_t offset_r = r * dim2;
                size_t k = offset_r;
                compressed_size = bufferSize;
                const void *input_ptr = &stemData.data()[k];
                if (pinned_in.ptr) {
                    CpuTimer tcopy;
                    tcopy.start();
                    memcpy(pinned_in.ptr, input_ptr, bufferSize);
                    input_copy_wall_stats.add(tcopy.stop_s());
                    input_ptr = pinned_in.ptr;
                }
                CpuTimer wall;
                wall.start();
                auto status = timed_mgard_compress(
                    compress_stats, 3, mgard_x::data_type::Float, data_shape_3d, tol, s,
                    mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                    compressed_size, config, true);
                compress_wall_stats.add(wall.stop_s());
                if (status != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                    return 2;
                }
                total_compressed += compressed_size;
                total_dataSize   += bufferSize;
                {
                    CpuTimer wall_d;
                    wall_d.start();
                    auto dstatus = timed_mgard_decompress(
                        decompress_stats, compressed_data, compressed_size, decompressed,
                        config, true);
                    decompress_wall_stats.add(wall_d.stop_s());
                    if (dstatus != mgard_x::compress_status_type::Success) {
                        std::cerr << "mgard_x::decompress failed: " << to_string(dstatus)
                                  << "\n";
                        return 3;
                    }
                }
                {
                    CpuTimer tmem;
                    tmem.start();
                    memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                    memcpy_wall_stats.add(tmem.stop_s());
                }
            }
        } else if (px >= 0) {
            size_t offset_r = px * dim2;
            size_t k = offset_r;
            compressed_size = bufferSize;
            const void *input_ptr = &stemData.data()[k];
            if (pinned_in.ptr) {
                CpuTimer tcopy;
                tcopy.start();
                memcpy(pinned_in.ptr, input_ptr, bufferSize);
                input_copy_wall_stats.add(tcopy.stop_s());
                input_ptr = pinned_in.ptr;
            }
            CpuTimer wall;
            wall.start();
            auto status = timed_mgard_compress(
                compress_stats, 3, mgard_x::data_type::Float, data_shape_3d, tol, s,
                mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                compressed_size, config, true);
            compress_wall_stats.add(wall.stop_s());
            if (status != mgard_x::compress_status_type::Success) {
                std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                return 2;
            }
            total_compressed += compressed_size;
            total_dataSize   += bufferSize;
            {
                CpuTimer wall_d;
                wall_d.start();
                auto dstatus = timed_mgard_decompress(
                    decompress_stats, compressed_data, compressed_size, decompressed, config,
                    true);
                decompress_wall_stats.add(wall_d.stop_s());
                if (dstatus != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::decompress failed: " << to_string(dstatus) << "\n";
                    return 3;
                }
            }
            {
                CpuTimer tmem;
                tmem.start();
                memcpy(&stemData_rct.data()[k], (float*)decompressed, bufferSize);
                memcpy_wall_stats.add(tmem.stop_s());
            }
        } else if (py >= 0) {
            std::vector<float> data_buf(256*256*256);
            for (size_t r=0; r<256; r++) {
                size_t offset_r = px * dim2;
                size_t k = offset_r;
                memcpy(&data_buf.data()[r*256], &stemData.data()[k], 256*sizeof(float));
            }
            const void *input_ptr = data_buf.data();
            if (pinned_in.ptr) {
                CpuTimer tcopy;
                tcopy.start();
                memcpy(pinned_in.ptr, input_ptr, bufferSize);
                input_copy_wall_stats.add(tcopy.stop_s());
                input_ptr = pinned_in.ptr;
            }
            CpuTimer wall;
            wall.start();
            auto status = timed_mgard_compress(
                compress_stats, 3, mgard_x::data_type::Float, data_shape_3d, tol, s,
                mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                compressed_size, config, true);
            compress_wall_stats.add(wall.stop_s());
            if (status != mgard_x::compress_status_type::Success) {
                std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                return 2;
            }
            total_compressed += compressed_size;
            total_dataSize   += bufferSize;
            {
                CpuTimer wall_d;
                wall_d.start();
                auto dstatus = timed_mgard_decompress(
                    decompress_stats, compressed_data, compressed_size, decompressed, config,
                    true);
                decompress_wall_stats.add(wall_d.stop_s());
                if (dstatus != mgard_x::compress_status_type::Success) {
                    std::cerr << "mgard_x::decompress failed: " << to_string(dstatus) << "\n";
                    return 3;
                }
            }
            for (size_t r=0; r<256; r++) {
                size_t offset_r = px * dim2;
                size_t k = offset_r;
                CpuTimer tmem;
                tmem.start();
                memcpy(&stemData_rct.data()[k], &((float*)decompressed)[r*256], 256*sizeof(float));
                memcpy_wall_stats.add(tmem.stop_s());
            }
        }
    } else if (ndim==4) {
        size_t nrows = 256 / r_interval;
        for (size_t r = 0; r < nrows; r++) {
            std::cout << "r = " << r << "\n";
            size_t offset_r = r * r_interval * dim2;
            compressed_size = bufferSize;
            const void *input_ptr = &stemData.data()[offset_r];
            if (pinned_in.ptr) {
                CpuTimer tcopy;
                tcopy.start();
                memcpy(pinned_in.ptr, input_ptr, bufferSize);
                input_copy_wall_stats.add(tcopy.stop_s());
                input_ptr = pinned_in.ptr;
            }
            CpuTimer wall;
            wall.start();
            auto status = timed_mgard_compress(
                compress_stats, 4, mgard_x::data_type::Float, data_shape_4d, tol, s,
                mgard_x::error_bound_type::ABS, input_ptr, compressed_data,
                compressed_size, config, true);
            compress_wall_stats.add(wall.stop_s());
            if (status != mgard_x::compress_status_type::Success) {
                std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
                return 2;
            }
            total_compressed += compressed_size;
            total_dataSize   += bufferSize;
            std::cout << "total data size = " << total_dataSize << ", total compressed size = " << total_compressed << "\n";
            //mgard_x::decompress(compressed_data, compressed_size, decompressed, config, true);
            //memcpy(&stemData_rct.data()[offset_r], (float*)decompressed, bufferSize);
        }
    } else if (ndim==0) { // compress the entire 4D data as a 3D data
        compressed_size = bufferSize;
        CpuTimer wall;
        wall.start();
        auto status = timed_mgard_compress(
            compress_stats, 3, mgard_x::data_type::Float, data_shape_3d_4d, tol, s,
            mgard_x::error_bound_type::ABS, &stemData.data()[0], compressed_data,
            compressed_size, config, true);
        compress_wall_stats.add(wall.stop_s());
        if (status != mgard_x::compress_status_type::Success) {
            std::cerr << "mgard_x::compress failed: " << to_string(status) << "\n";
            return 2;
        }
        total_compressed += compressed_size;
        total_dataSize   += bufferSize;
        // mgard_x::decompress(compressed_data, bufferSize, decompressed, config, true);
        // memcpy(&stemData_rct.data()[0], (float*)decompressed, bufferSize);
    }

    if (compress_stats.calls > 0) {
        const double ms_to_s = 1e-3;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mgard_x::compress GPU timing (s): calls=" << compress_stats.calls
                  << ", total=" << (compress_stats.total_ms * ms_to_s)
                  << ", avg=" << ((compress_stats.total_ms / (double)compress_stats.calls) * ms_to_s)
                  << ", min=" << (compress_stats.min_ms * ms_to_s)
                  << ", max=" << (compress_stats.max_ms * ms_to_s)
                  << "\n";
    }

    if (decompress_stats.calls > 0) {
        const double ms_to_s = 1e-3;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mgard_x::decompress GPU timing (s): calls=" << decompress_stats.calls
                  << ", total=" << (decompress_stats.total_ms * ms_to_s)
                  << ", avg=" << ((decompress_stats.total_ms / (double)decompress_stats.calls) * ms_to_s)
                  << ", min=" << (decompress_stats.min_ms * ms_to_s)
                  << ", max=" << (decompress_stats.max_ms * ms_to_s)
                  << "\n";
    }

    if (compress_wall_stats.calls > 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mgard_x::compress wall timing (s): calls=" << compress_wall_stats.calls
                  << ", total=" << compress_wall_stats.total_s
                  << ", avg=" << (compress_wall_stats.total_s / (double)compress_wall_stats.calls)
                  << ", min=" << compress_wall_stats.min_s
                  << ", max=" << compress_wall_stats.max_s
                  << "\n";
    }

    if (decompress_wall_stats.calls > 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mgard_x::decompress wall timing (s): calls=" << decompress_wall_stats.calls
                  << ", total=" << decompress_wall_stats.total_s
                  << ", avg=" << (decompress_wall_stats.total_s / (double)decompress_wall_stats.calls)
                  << ", min=" << decompress_wall_stats.min_s
                  << ", max=" << decompress_wall_stats.max_s
                  << "\n";
    }

    if (memcpy_wall_stats.calls > 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "memcpy wall timing (s): calls=" << memcpy_wall_stats.calls
                  << ", total=" << memcpy_wall_stats.total_s
                  << ", avg=" << (memcpy_wall_stats.total_s / (double)memcpy_wall_stats.calls)
                  << ", min=" << memcpy_wall_stats.min_s
                  << ", max=" << memcpy_wall_stats.max_s
                  << "\n";
    }

    if (input_copy_wall_stats.calls > 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "input memcpy(to pinned) wall timing (s): calls="
                  << input_copy_wall_stats.calls << ", total="
                  << input_copy_wall_stats.total_s << ", avg="
                  << (input_copy_wall_stats.total_s /
                      (double)input_copy_wall_stats.calls)
                  << ", min=" << input_copy_wall_stats.min_s
                  << ", max=" << input_copy_wall_stats.max_s << "\n";
    }

    const double total_s = total_timer.stop_s();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Overall wall time (s): total=" << total_s
              << ", file_read=" << io_read_s
                            << ", alloc_stem=" << alloc_stem_s
                            << ", alloc_rct=" << alloc_rct_s
                            << ", alloc_workbuf=" << alloc_workbuf_s
            //   << ", minmax=" << minmax_s
              << "\n";

        if (pinned_setup_wall_stats.calls > 0) {
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "pinned buffer setup wall timing (s): calls="
                                    << pinned_setup_wall_stats.calls << ", total="
                                    << pinned_setup_wall_stats.total_s << ", avg="
                                    << (pinned_setup_wall_stats.total_s /
                                            (double)pinned_setup_wall_stats.calls)
                                    << ", min=" << pinned_setup_wall_stats.min_s
                                    << ", max=" << pinned_setup_wall_stats.max_s << "\n";
        }

    // double rmse = 0.0, diff = 0;
    // for (size_t m=0; m<nData; m++) {
    //     diff = (double)stemData_rct.data()[m] - (double)stemData.data()[m];
    //     rmse += diff * diff;
    // }

    // rmse = std::sqrt(rmse / (float)nData);
    // std::cout << "rmse = " << rmse << ", NRSEM = " << rmse / (maxv-minv) <<"\n"; 
    // free(decompressed);
    // free(compressed_data);
    std::cout << "Compression ratios = " << (float)total_dataSize / (float)total_compressed << "\n";
    // std::cout << "Compression ratios = " << (float)total_dataSize / (float)total_compressed << ", rmse = " << rmse << ", NRSEM = " << rmse / (maxv-minv) <<"\n";  
    // fp = fopen((filename + ".mgr").c_str(),"wb");
    // fwrite(stemData_rct.data(), sizeof(float), nData, fp);
    // fclose(fp);
    
    return 0;
}

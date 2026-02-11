#pragma once

#include <cstdint>

// Common Marlin constants shared across CUDA/HIP codepaths.
constexpr int kDefaultThreads = 256;
constexpr int kPipeStages = 4;
constexpr int kMinThreadN = 64;
constexpr int kMinThreadK = 64;
constexpr int kMaxThreadN = 256;
constexpr int kTileSize = 16;
constexpr int kMaxPar = 16;

constexpr int kRepackStages = 8;
constexpr int kRepackThreads = 256;
constexpr int kTileKSize = kTileSize;
constexpr int kTileNSize = kTileKSize * 4;

constexpr int kATilePad = 1;
constexpr int kATileStride = kTileSize + kATilePad;
constexpr int kATileElems = kTileSize * kATileStride;
constexpr int kTileElems = kTileSize * kTileSize;

constexpr int kBTileCols = kTileSize;
constexpr int kBTileRows = kTileSize;
constexpr int kBTilePad = 1;
constexpr int kBTileStride = kBTileCols + kBTilePad;
constexpr int kBTileWords = kBTileRows * kBTileCols;
constexpr int kBTileWordsLds = kBTileRows * kBTileStride;

constexpr int ERR_PROB_SHAPE = 1;
constexpr int ERR_KERN_SHAPE = 2;

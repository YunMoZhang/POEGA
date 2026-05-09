// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BITMAP_H_
#define BITMAP_H_

#include <algorithm>
#include <cinttypes>
#include "assert.h"
#include "gpu_error_check.hpp"


class Bitmap {
 public:
  explicit Bitmap(size_t size) {
    uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
    start_ = new uint64_t[num_words];
    end_ = start_ + num_words;
    size_ = size;
    // onHost = true;
    wordsLen = num_words;
  }

  Bitmap() {
    start_ = nullptr;
    end_ = nullptr;
    size_ = 0;
  }

  void alloc(size_t size){
    assert(start_ == nullptr);
    uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
    start_ = new uint64_t[num_words];
    end_ = start_ + num_words;
    size_ = size;
    wordsLen = num_words;
    // onHost = true;
  }

  // void allocHost(size_t size){
  //   assert(start_ == nullptr);
  //   uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
  //   gpuErrorcheck(cudaMallocHost(&start_, (num_words) * sizeof(uint64_t)));
  //   end_ = start_ + num_words;
  //   size_ = size;
  // }

  void allocDevice(size_t size){
    assert(start_ == nullptr);
    uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
    gpuErrorcheck(cudaMalloc(&start_, num_words * sizeof(uint64_t)));
    end_ = start_ + num_words;
    size_ = size;
    wordsLen = num_words;
    // onHost = false;
  }

  void copy(Bitmap &other){
    assert(start_ == nullptr);
    uint64_t num_words = (other.size_ + kBitsPerWord - 1) / kBitsPerWord;
    start_ = new uint64_t[num_words];
    size_ = other.size_;
    end_ = start_ + num_words;
    memcpy(start_, other.start_, num_words * sizeof(uint64_t));
    wordsLen = other.wordsLen;
  }

  ~Bitmap() {
    delete[] start_;
  }

  void reset() {
    std::fill(start_, end_, 0);
  }

  void set() { // added by yunmo
    uint64_t all_one = ~(0l);
    std::fill(start_, end_, all_one);
  }

  void set_bit(size_t pos) {
    start_[word_offset(pos)] |= ((uint64_t) 1l << bit_offset(pos));
  }

  // void set_bit_atomic(size_t pos) {
  //   uint64_t old_val, new_val;
  //   do {
  //     old_val = start_[word_offset(pos)];
  //     new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
  //   } while (!compare_and_swap(start_[word_offset(pos)], old_val, new_val));
  // }

  __host__ __device__ bool get_bit(size_t pos) const {
    return (start_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  void reset_bit(size_t pos){ // added by yunmo
    start_[word_offset(pos)] &= ~((uint64_t) 1l << bit_offset(pos));
  }

  void swap(Bitmap &other) {
    std::swap(start_, other.start_);
    std::swap(end_, other.end_);
    std::swap(size_, other.size_);
  }

  uint64_t * begin() const{
    return start_;
  }

  uint64_t * end() const{
    return end_;
  }

  size_t words_len() const{
    return wordsLen;
  }

  // __host__ __device__ bool isOnHost() const{
  //   return onHost;
  // }

 private:
  uint64_t *start_;
  uint64_t *end_;
  size_t size_;
  size_t wordsLen;
  // bool onHost;

  static const uint64_t kBitsPerWord = 64;
  __host__ __device__ static uint64_t word_offset(size_t n) { return n / kBitsPerWord; }
  __host__ __device__ static uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }
};

#endif  // BITMAP_H_
// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef PVECTOR_H_
#define PVECTOR_H_

#include <algorithm>
#include "omp.h"

template <typename T_>
class pvector {
 public:
  typedef T_* iterator;

  pvector() : arr_(nullptr), begin_(nullptr), end_(nullptr), end_capacity_(nullptr) {}

  explicit pvector(size_t num_elements) {
    arr_ = new T_[num_elements];
    begin_ = arr_;
    end_ = arr_ + num_elements;
    end_capacity_ = end_;
  }

  pvector(size_t num_elements, T_ init_val) : pvector(num_elements) {
    fill(init_val);
  }

  pvector(iterator copy_begin, iterator copy_end)
      : pvector(copy_end - copy_begin) {
    #pragma omp parallel for
    for (size_t i=0; i < capacity(); i++)
      arr_[i] = copy_begin[i];
  }

  // don't want this to be copied, too much data to move
  pvector(const pvector &other) = delete;

  // prefer move because too much data to copy
  pvector(pvector &&other)
      : arr_(other.arr_), begin_(other.begin_), end_(other.end_),
        end_capacity_(other.end_capacity_) {
    other.arr_ = nullptr;
    other.begin_ = nullptr;
    other.end_ = nullptr;
    other.end_capacity_ = nullptr;
  }

  // want move assignment
  pvector& operator= (pvector &&other) {
    if (this != &other) {
      ReleaseResources();
      arr_ = other.arr_;
      begin_ = other.begin_;
      end_ = other.end_;
      end_capacity_ = other.end_capacity_;
      other.arr_ = nullptr;
      other.begin_ = nullptr;
      other.end_ = nullptr;
      other.end_capacity_ = nullptr;
    }
    return *this;
  }

  void ReleaseResources(){
    if (arr_ != nullptr) {
      delete[] arr_;
      arr_ = nullptr;
    }
  }

  ~pvector() {
    if(arr_ != nullptr)
      ReleaseResources();
  }

  void alloc(size_t num_elements) {
    if(arr_ != nullptr) delete [] arr_;
    arr_ = new T_[num_elements];
    begin_ = arr_;
    end_ = arr_ + num_elements;
    end_capacity_ = end_;
  }

  // not thread-safe
  void reserve(size_t num_elements) {
    if (num_elements > capacity()) {
      T_ *new_range = new T_[num_elements];
      size_t invalid_front = begin_ - arr_;
      #pragma omp parallel for
      for (size_t i=invalid_front; i < size(); i++)
        new_range[i - invalid_front] = arr_[i];
      end_ = new_range + size();
      delete[] arr_;
      arr_ = new_range;
      begin_ = arr_ + invalid_front;
      end_capacity_ = arr_ + num_elements;
    }
  }

  // prevents internal storage from being freed when this pvector is desctructed
  // - used by Builder to reuse an EdgeList's space for in-place graph building
//   void leak() {
//     start_ = nullptr;
//   }

  bool empty() {
    return end_ == begin_;
  }

  void clear() {
    begin_ = arr_;
    end_ = begin_;
  }

  void resize(size_t num_elements) {
    reserve(num_elements);
    // end_ = arr_ + num_elements;
  }

  T_& operator[](size_t n) {
    return begin_[n];
  }

  const T_& operator[](size_t n) const {
    return begin_[n];
  }

  void push_back(T_ val) {
    if (size() == capacity()) {
      size_t new_size = capacity() == 0 ? 1 : capacity() * growth_factor;
      reserve(new_size);
    }
    *end_ = val;
    end_++;
  }

  void pop_front(size_t pop_num = 1) {
    begin_ += pop_num;
  }

  void fill(T_ init_val) {
    #pragma omp parallel for
    for (T_* ptr=arr_; ptr < end_; ptr++)
      *ptr = init_val;
  }

  size_t capacity() const {
    return end_capacity_ - arr_;
  }

  size_t size() const {
    return end_ - begin_;
  }

  iterator begin() const {
    return begin_;
  }

  iterator end() const {
    return end_;
  }

  T_* data() const {
    return begin_;
  }

  T_* arr() const{
    return arr_;
  }

  void swap(pvector &other) {
    std::swap(arr_, other.arr_);
    std::swap(begin_, other.begin_);
    std::swap(end_, other.end_);
    std::swap(end_capacity_, other.end_capacity_);
  }


 private:
  T_* arr_;
  T_* begin_;
  T_* end_;
  T_* end_capacity_;
  static const size_t growth_factor = 2;
};

#endif  // PVECTOR_H_
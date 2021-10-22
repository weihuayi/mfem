// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR
#define MFEM_TENSOR

#include "tensor_traits.hpp"
#include "containers/containers.hpp"
#include "layouts/layouts.hpp"
#include "utilities/foreach.hpp"

namespace mfem
{

/** A tensor class
    @a Container is the type of data container, they can either be statically or
       dynamically allocated,
    @a Layout is a class that represents the data layout
       There is two main sub-categories of Layout, Static and Dynamic layouts.
       Dynamic Layout have the following signature:
       template <int Rank>,
       Static Layout have the following signature:
       template <int... Sizes>,
       where Sizes... is the list of the sizes of the dimensions of the Tensor.
   */
template <typename Container,
          typename Layout>
class Tensor: public Container, public Layout
{
public:
   using T = get_container_type<Container>;
   using container = Container;
   using layout = Layout;

   /// Default Constructor
   MFEM_HOST_DEVICE
   Tensor() : Container(), Layout() { }

   /// Main Constructors
   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(int size0, Sizes... sizes)
   : Container(size0,sizes...), Layout(size0,sizes...) { }

   template <typename... Sizes> MFEM_HOST_DEVICE
   Tensor(T* ptr, Sizes... sizes)
   : Container(ptr), Layout(sizes...) { }

   /// Utility Constructors
   MFEM_HOST_DEVICE
   Tensor(Layout index): Container(), Layout(index) { }

   MFEM_HOST_DEVICE
   Tensor(Container data, Layout index): Container(data), Layout(index) { }

   /// Copy Constructors
   MFEM_HOST_DEVICE
   Tensor(const Tensor &rhs): Container(rhs), Layout(rhs) { }

   template <typename OtherTensor> MFEM_HOST_DEVICE
   Tensor(const OtherTensor &rhs): Container(), Layout(rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
   }

   /// Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   T& operator()(Idx... args)
   {
      static_assert(get_layout_rank<Layout> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Const Accessor
   template <typename... Idx> MFEM_HOST_DEVICE inline
   const T& operator()(Idx... args) const
   {
      static_assert(get_layout_rank<Layout> == sizeof...(Idx),
                    "Wrong number of indices");
      return this->operator[]( this->index(args...) );
   }

   /// Initialization of a Tensor to a constant value.
   MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const T &val)
   {
      ForallDims<Tensor>::Apply(*this, [&](auto... idx)
      {
         (*this)(idx...) = val;
      });
      return *this;
   }

   /// operator=, compatible with other types of Tensors
   template <typename OtherTensor> MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) = rhs(idx...);
      });
      return *this;
   }

   /// operator+=, compatible with other types of Tensors
   template <typename OtherTensor> MFEM_HOST_DEVICE inline
   Tensor<Container,Layout>& operator+=(const OtherTensor &rhs)
   {
      ForallDims<Tensor>::ApplyBinOp(*this, rhs, [&](auto... idx)
      {
         (*this)(idx...) += rhs(idx...);
      });
      return *this;
   }

   /// Lazy accessor for the sub-Tensor extracted from idx in Nth dimension.
   template <int N> MFEM_HOST_DEVICE inline
   auto Get(int idx)
   {
      static_assert(N>=0 && N<get_layout_rank<Layout>,
         "Cannot access this dimension with Get");
      using C = ViewContainer<T,Container>;
      using L = RestrictedLayout<N,Layout>;
      using RestrictedTensor = Tensor<C,L>;
      C data(*this);
      L layout(idx,*this);
      return RestrictedTensor(data,layout);
   }

   template <int N> MFEM_HOST_DEVICE inline
   auto Get(int idx) const
   {
      static_assert(N>=0 && N<get_layout_rank<Layout>,
         "Cannot access this dimension with Get");
      using C = ConstViewContainer<T,Container>;
      using L = RestrictedLayout<N,Layout>;
      using RestrictedTensor = Tensor<C,L>;
      C data(*this);
      L layout(idx,*this);
      return RestrictedTensor(data,layout);
   }
};

} // namespace mfem

#endif // MFEM_TENSOR
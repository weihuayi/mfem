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

#ifndef MFEM_TENSOR_DOF
#define MFEM_TENSOR_DOF

#include "dof_impl.hpp"

namespace mfem
{

/// A class to encapsulate degrees of freedom in a Tensor.
template <typename DofTensor>
class DegreesOfFreedom
: public DofTensor
{
public:
   template <typename... Sizes> MFEM_HOST_DEVICE
   DegreesOfFreedom(double *x, Sizes... sizes)
   : DofTensor(x, sizes...)
   {
      // TODO static asserts Config values
   }

   /// Returns a Tensor corresponding to the DoFs of element e
   MFEM_HOST_DEVICE inline
   auto operator()(int e) const
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e); // TODO batchsize so +tidz or something?
   }

   MFEM_HOST_DEVICE inline
   auto operator()(int e)
   {
      constexpr int Rank = get_tensor_rank<DofTensor>;
      return this->template Get<Rank-1>(e);
   }
};

/// Functor to represent degrees of freedom
template <int VDim, typename Config>
auto MakeDoFs(Config &config, double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<DeviceContainer<double>,Layout>;
   using Dof = DegreesOfFreedom<DofTensor>;
   return InitDof<Dof,IsTensor,Dim,VDim>::make(x,config.dofs,ne);
}

template <int VDim, typename Config>
auto MakeDoFs(Config &config, const double *x, int ne)
{
   constexpr int Dim = get_config_dim<Config>;
   constexpr bool IsTensor = is_tensor_config<Config>;
   constexpr int Dofs = get_config_dofs<Config>;
   using Layout = get_dof_layout<IsTensor,Dofs,Dim,VDim>;
   using DofTensor = Tensor<ReadContainer<double>,Layout>;
   using Dof = DegreesOfFreedom<DofTensor>;
   return InitDof<Dof,IsTensor,Dim,VDim>::
      make(const_cast<double*>(x),config.dofs,ne);
}

} // mfem namespace

#endif // MFEM_TENSOR_DOF
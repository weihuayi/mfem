// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_BACKENDS_PA_ENGINE_HPP
#define MFEM_BACKENDS_PA_ENGINE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "../base/backend.hpp"
#include "util.hpp"
// #include "layout.hpp"
// #include "pa_array.hpp"
// #include "vector.hpp"
// #include "pa_fespace.hpp"

namespace mfem
{

namespace pa
{

template <Location Device>
class PAEngine : public mfem::Engine
{
protected:
   //
   // Inherited fields
   //
   // mfem::Backend *backend;
#ifdef MFEM_USE_MPI
   // MPI_Comm comm;
#endif
   // int num_mem_res;
   // int num_workers;
   // MemoryResource **memory_resources;
   // double *workers_weights;
   // int *workers_mem_res;

   void Init(const std::string &engine_spec);

public:
   PAEngine();
   PAEngine(const std::string &engine_spec);

#ifdef MFEM_USE_MPI
   PAEngine(MPI_Comm comm, const std::string &engine_spec);
#endif

   virtual ~PAEngine() { }

   /**
       @name Virtual interface: finite element data structures and algorithms
    */
   ///@{

   virtual DLayout MakeLayout(std::size_t size) const;
   virtual DLayout MakeLayout(const mfem::Array<std::size_t> &offsets) const;

   virtual DArray MakeArray(PLayout &layout, std::size_t item_size) const;

   virtual DVector MakeVector(PLayout &layout,
                              int type_id = ScalarId<double>::value) const;

   virtual DFiniteElementSpace MakeFESpace(mfem::FiniteElementSpace &
                                           fespace) const;

   virtual DBilinearForm MakeBilinearForm(mfem::BilinearForm &bf) const;

   /// FIXME - What will the actual parameters be?
   virtual void AssembleLinearForm(LinearForm &l_form) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const MixedBilinearForm &mbl_form) const;

   /// FIXME - What will the actual parameters be?
   virtual mfem::Operator *MakeOperator(const NonlinearForm &nl_form) const;

   ///@}
   // End: Virtual interface
};

mfem::Engine* createEngine(const std::string& engine_spec);

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_ENGINE_HPP

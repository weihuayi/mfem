// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#define MFEM_DEBUG_COLOR 86
#include "general/debug.hpp"

#include "unit_tests.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

namespace quadrature_interpolator_face
{


static void QuadratureInterpolatorFaceTest(const char *file, int p, int q)
{
   Mesh mesh(file);
   mesh.EnsureNodes();
   const int dim = mesh.Dimension();
   constexpr double a = 1.0, b = -0.5;
   Geometry::Type geom = mesh.GetElementGeometry(0);
   VectorFunctionCoefficient v(dim, [](const Vector &x, Vector &v)
   {
      switch (x.Size())
      {
         case 1: v(0) = 1.0; break;
         case 2: v(0) = x(1); v(1) = -x(0); break;
         case 3: v(0) = x(1); v(1) = -x(0); v(2) = x(0); break;
      }
   });

   L2_FECollection fec(p, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   BilinearForm k_tst(&fes);
   BilinearForm k_ref(&fes);

   // Don't use a special integration rule if q == 0

   auto AddConvectionIntegrators = [&](BilinearForm &bf)
   {
      auto AddNewDGTraceInteg = [&]()
      {
         auto dgti = new DGTraceIntegrator(v, a, b);
         dbg("p:%d, 2p+q:%d", p, 2*p+q);
         return (dgti->SetIntRule(&IntRules.Get(geom, 2*p+q)), dgti);
      };
      bf.AddDomainIntegrator(new ConvectionIntegrator(v, -a));
      bf.AddInteriorFaceIntegrator(new TransposeIntegrator(AddNewDGTraceInteg()));
      bf.AddBdrFaceIntegrator(new TransposeIntegrator(AddNewDGTraceInteg()));
   };

   AddConvectionIntegrators(k_ref);
   k_ref.SetAssemblyLevel(AssemblyLevel::FULL);
   k_ref.Assemble();
   k_ref.Finalize();
   k_ref.SpMat().EnsureMultTranspose();

   AddConvectionIntegrators(k_tst);
   k_tst.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_tst.Assemble();

   GridFunction x(&fes), y_ref(&fes), y_tst(&fes);
   x.Randomize(1);

   k_ref.Mult(x, y_ref);
   k_tst.Mult(x, y_tst);
   y_tst -= y_ref;
   REQUIRE(y_tst.Norml2() < 1.e-12);

   // Test MultTranspose
   k_ref.MultTranspose(x, y_ref);
   k_tst.MultTranspose(x, y_tst);
   y_tst -= y_ref;
   REQUIRE(y_tst.Norml2() < 1.e-12);
   dbg();
}

TEST_CASE("QuadratureInterpolatorFace",
          "[QuadratureInterpolatorFace]"
          "[QuadratureInterpolator]"
          "[CUDA]")
{
   //const bool all_tests = launch_all_non_regression_tests;
   //auto q = !all_tests ? 0 : GENERATE(0, 1, 3);
   //auto p = !all_tests ? GENERATE(2, 3) : GENERATE(1, 2, 3);
   auto p = GENERATE(0);
   auto q = GENERATE(0);

   QuadratureInterpolatorFaceTest("../../data/star-q3.mesh", p, q);
   QuadratureInterpolatorFaceTest("../../data/fichera-q3.mesh", p, q);
} // QuadratureInterpolatorFace test case

} // quadrature_interpolator_face namespace

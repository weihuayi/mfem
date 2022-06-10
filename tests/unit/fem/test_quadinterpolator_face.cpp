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
#include <string>

using namespace mfem;

namespace quadrature_interpolator_face
{

static void ScalarQuadInterpFaceTest(std::string base, std::string qn, int p)
{
   constexpr int VDIM = 1;

   std::string path("../../data/");
   std::string file = path + base + qn + ".mesh";
   Mesh mesh(file.c_str(),1,1);
   mesh.EnsureNodes();
   const int dim = mesh.Dimension();

   L2_FECollection fec(p, dim, BasisType::GaussLobatto);

   FiniteElementSpace sfes(&mesh, &fec, VDIM);
   GridFunction s(&sfes);
   s.Randomize();

   const FaceType type = FaceType::Interior;
   const FaceRestriction *face_restr = sfes.GetFaceRestriction(
                                          ElementDofOrdering::LEXICOGRAPHIC,
                                          type,
                                          L2FaceValues::SingleValued );
   Vector e_vec(face_restr->Height());
   face_restr->Mult(s, e_vec);

   FaceElementTransformations *T = mesh.GetFaceElementTransformations(0);
   const FiniteElement &el = *sfes.GetTraceElement(0, mesh.GetFaceBaseGeometry(0));
   const int q_order = T->Elem1->OrderW() + 2*el.GetOrder();
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), q_order);

   const FaceQuadratureInterpolator *qi =
      sfes.GetFaceQuadratureInterpolator(ir, type);

   const int NQ = ir.GetNPoints();
   const int NF = sfes.GetNFbyType(type);
   Vector q_val(VDIM*NQ*NF), q_der(VDIM*VDIM*NQ*NF),
          q_det(NQ*NF), q_nor(VDIM*NQ*NF);
   unsigned eval_flags = 0;
   eval_flags |= FaceQuadratureInterpolator::VALUES;
   eval_flags |= FaceQuadratureInterpolator::NORMALS;
   eval_flags |= FaceQuadratureInterpolator::DERIVATIVES;
   qi->Mult(e_vec, eval_flags, q_val, q_der, q_det, q_nor);
   dbg("e_vec: %.15e",e_vec*e_vec);
   dbg("q_val: %.15e",q_val*q_val);
}

static void VectorQuadInterpFaceTest(std::string base, std::string qn, int p)
{
   std::string path("../../data/");
   std::string file = path + base + qn + ".mesh";
   Mesh mesh(file.c_str(),1,1);
   mesh.EnsureNodes();
   const int dim = mesh.Dimension();

   constexpr double a = 1.0, b = -0.5;
   VectorFunctionCoefficient Q(dim, [](const Vector &x, Vector &v)
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

   auto AddFaceIntegrator = [&](BilinearForm &bf)
   {
      bf.AddInteriorFaceIntegrator(
         new TransposeIntegrator(
            new DGTraceIntegrator(Q,a,b)));
   };

   AddFaceIntegrator(k_ref);
   k_ref.SetAssemblyLevel(AssemblyLevel::FULL);
   k_ref.Assemble();
   k_ref.Finalize();
   k_ref.SpMat().EnsureMultTranspose();

   AddFaceIntegrator(k_tst);
   k_tst.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_tst.Assemble();

   GridFunction x(&fes), y_ref(&fes), y_tst(&fes);
   x.Randomize();

   k_ref.Mult(x, y_ref);
   k_tst.Mult(x, y_tst);
   y_tst -= y_ref;
   REQUIRE(y_tst.Norml2() < 1.e-12);
}

TEST_CASE("QuadratureInterpolatorFace", "[QuadratureInterpolatorFace][CUDA]")
{
   auto base = GENERATE("star", "fichera");
   auto qn = GENERATE("", "-q2", "-q3");
   auto p = GENERATE(1,2);
   VectorQuadInterpFaceTest(base, qn, p);
   ScalarQuadInterpFaceTest(base, qn, p);
}

} // quadrature_interpolator_face
